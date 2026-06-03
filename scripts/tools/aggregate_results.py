import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "outputs"
EVAL_METRICS = [
    "accuracy",
    "top5_accuracy",
    "balanced_accuracy_present",
    "macro_f1_present",
    "macro_f1_all_182",
]


def read_text(path):
    return path.read_text(encoding="utf-8", errors="replace")


def find_first(pattern, text):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_eval_metrics(text):
    metrics = {}
    for name, value in re.findall(r"^\*\s+([A-Za-z0-9_]+):\s+([0-9.]+)%?\s*$", text, flags=re.MULTILINE):
        if name in EVAL_METRICS:
            metrics[name] = value
    return metrics


def normalize_eval_name(name):
    if name.startswith("eval_"):
        name = name[len("eval_"):]
    name = re.sub(r"epoch_(\d+)", r"epoch\1", name)
    return name


def parse_path_metadata(path):
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    eval_index = next((i for i, p in enumerate(parts) if p.startswith("eval_")), None)
    shot = ""
    for p in parts:
        m = re.fullmatch(r"shot_(.+)", p)
        if m:
            shot = "-1" if m.group(1) == "full" else m.group(1)
            break

    eval_name = normalize_eval_name(parts[eval_index]) if eval_index is not None else ""
    run_parts = parts[:eval_index] if eval_index is not None else parts[:-1]
    run_parts = [p for p in run_parts if not re.fullmatch(r"shot_.+", p)]
    run_id = "/".join(run_parts)
    return rel.as_posix(), run_id, shot, eval_name


def parse_train_path_metadata(path):
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    train_index = next((i for i, p in enumerate(parts) if p == "train"), None)
    shot = ""
    for p in parts:
        m = re.fullmatch(r"shot_(.+)", p)
        if m:
            shot = "-1" if m.group(1) == "full" else m.group(1)
            break
    run_parts = parts[:train_index] if train_index is not None else parts[:-1]
    run_parts = [p for p in run_parts if not re.fullmatch(r"shot_.+", p)]
    return rel.as_posix(), "/".join(run_parts), shot


def infer_status(path, run_id):
    text = path.as_posix().lower()
    if "smoke" in text:
        return "smoke"
    if "coop_lora_siglip_iwildcam_vitb16_full_10ep_r4_l8/" in text:
        return "invalid_resume"
    if "coop_lora_iwildcam_vitb16_full_10ep_ablation/r8_l8/" in text:
        return "invalid_resume"
    if "wildcoop" in text:
        return "historical"
    return "trusted"


def parse_eval_logs():
    rows = []
    for path in sorted(ROOT.rglob("log.txt")):
        if not any(part.startswith("eval_") for part in path.parts):
            continue
        text = read_text(path)
        metrics = parse_eval_metrics(text)
        if not metrics:
            continue

        rel, run_id, shot, eval_name = parse_path_metadata(path)
        row = {
            "source_file": rel,
            "run_id": run_id,
            "eval_name": eval_name,
            "shot": shot or find_first(r"^\s+NUM_SHOTS:\s*(.*)$", text),
            "seed": find_first(r"^seed:\s*(.*)$", text),
            "trainer": find_first(r"^trainer:\s*(.*)$", text),
            "backbone": find_first(r"^\s+NAME:\s*(ViT-[^\r\n]+)$", text),
            "config_file": find_first(r"^config_file:\s*(.*)$", text),
            "load_epoch": find_first(r"^load_epoch:\s*(.*)$", text),
            "status": infer_status(path.relative_to(ROOT), run_id),
        }
        for metric in EVAL_METRICS:
            row[metric] = metrics.get(metric, "")
        rows.append(row)
    rows.sort(key=lambda r: (r["status"], r["run_id"], r["eval_name"], r["source_file"]))
    return rows


def parse_train_logs_without_eval(eval_rows):
    eval_run_ids = {row["run_id"] for row in eval_rows}
    rows = []
    for path in sorted(ROOT.rglob("train/log.txt")):
        rel, run_id, shot = parse_train_path_metadata(path)
        if run_id in eval_run_ids:
            continue
        text = read_text(path)
        epochs = re.findall(r"epoch \[(\d+)/(\d+)\]", text)
        last_epoch = epochs[-1][0] if epochs else ""
        max_epoch = epochs[-1][1] if epochs else find_first(r"^\s+MAX_EPOCH:\s*(.*)$", text)
        rows.append(
            {
                "source_file": rel,
                "run_id": run_id,
                "shot": shot or find_first(r"^\s+NUM_SHOTS:\s*(.*)$", text),
                "seed": find_first(r"^seed:\s*(.*)$", text),
                "trainer": find_first(r"^trainer:\s*(.*)$", text),
                "backbone": find_first(r"^\s+NAME:\s*(ViT-[^\r\n]+)$", text),
                "last_epoch_seen": last_epoch,
                "max_epoch": max_epoch,
                "finished_training": "yes" if "Finish training" in text else "no",
                "status": infer_status(path.relative_to(ROOT), run_id),
            }
        )
    rows.sort(key=lambda r: (r["status"], r["run_id"], r["source_file"]))
    return rows


def is_metric_csv(path):
    name = path.name.lower()
    rel = path.relative_to(ROOT).as_posix().lower()
    if "per_class" in name or "iwildcam_per_class" in name:
        return False
    if "train_history" in rel or "train_epoch_summary" in rel:
        return False
    if rel.startswith("eda/"):
        return False
    return any(token in name for token in ["summary", "metrics", "comparison", "mean_std", "vs_"])


def parse_metric_csvs():
    records = []
    fieldnames = set(["source_file", "source_row"])
    for path in sorted(ROOT.rglob("*.csv")):
        if not is_metric_csv(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                for idx, row in enumerate(reader, start=2):
                    record = {"source_file": rel, "source_row": str(idx)}
                    for key, value in row.items():
                        clean_key = key if key is not None else "unknown"
                        record[clean_key] = value
                        fieldnames.add(clean_key)
                    records.append(record)
        except UnicodeDecodeError:
            continue
    ordered = ["source_file", "source_row"] + sorted(k for k in fieldnames if k not in {"source_file", "source_row"})
    return records, ordered


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def markdown_table(rows, columns):
    if not rows:
        return "(none)\n"
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def best_rows(rows, eval_contains, limit=20, statuses=("trusted",)):
    candidates = []
    for row in rows:
        if row.get("status") not in statuses:
            continue
        if eval_contains not in row.get("eval_name", ""):
            continue
        score = float_or_none(row.get("macro_f1_present"))
        if score is None:
            continue
        candidates.append(row)
    candidates.sort(key=lambda r: float(r["macro_f1_present"]), reverse=True)
    return candidates[:limit]


def write_markdown(path, eval_rows, csv_rows, train_only_rows):
    trusted = [r for r in eval_rows if r["status"] == "trusted"]
    nontrusted = [r for r in eval_rows if r["status"] != "trusted"]
    per_class_files = list(ROOT.rglob("iwildcam_per_class_eval1.csv")) + list(ROOT.rglob("*per_class.csv"))
    train_histories = list(ROOT.rglob("train_history.csv")) + list(ROOT.rglob("train_epoch_summary.csv"))

    top_cols = [
        ("run_id", "run_id"),
        ("eval", "eval_name"),
        ("trainer", "trainer"),
        ("shot", "shot"),
        ("seed", "seed"),
        ("acc", "accuracy"),
        ("bal_acc", "balanced_accuracy_present"),
        ("macro_f1", "macro_f1_present"),
        ("all182", "macro_f1_all_182"),
        ("status", "status"),
    ]
    all_cols = [
        ("status", "status"),
        ("run_id", "run_id"),
        ("eval", "eval_name"),
        ("trainer", "trainer"),
        ("shot", "shot"),
        ("seed", "seed"),
        ("epoch", "load_epoch"),
        ("acc", "accuracy"),
        ("top5", "top5_accuracy"),
        ("bal_acc", "balanced_accuracy_present"),
        ("macro_f1", "macro_f1_present"),
        ("all182", "macro_f1_all_182"),
    ]

    lines = []
    lines.append("# All DG-iWildCam Experiment Results\n")
    lines.append("Primary metric: `macro_f1_present` in percent. Generated from every aggregate eval `log.txt` plus metric CSVs under `outputs`.\n")
    lines.append("## Generated Artifacts\n")
    lines.append(f"- Eval-log aggregate rows: `{len(eval_rows)}` in `outputs/all_eval_log_results.csv`\n")
    lines.append(f"- Metric CSV rows: `{len(csv_rows)}` in `outputs/all_metric_csv_rows.csv`\n")
    lines.append(f"- Per-class CSV files found but not expanded here: `{len(per_class_files)}`\n")
    lines.append(f"- Train-history CSV files found but not expanded here: `{len(train_histories)}`\n")
    lines.append(f"- Train logs without aggregate eval rows: `{len(train_only_rows)}` in `outputs/train_logs_without_eval.csv`\n")
    lines.append("- Exhaustive aggregate eval rows are included below; source paths are preserved in the CSV.\n")

    lines.append("## Best Trusted OOD-Test Rows\n")
    lines.append(markdown_table(best_rows(eval_rows, "ood_test", limit=20), top_cols))

    lines.append("## Best Trusted OOD-Val Rows\n")
    lines.append(markdown_table(best_rows(eval_rows, "ood_val", limit=30), top_cols))

    lines.append("## All Trusted Aggregate Eval Rows\n")
    lines.append(markdown_table(trusted, all_cols))

    lines.append("## Historical, Smoke, Or Invalid Rows\n")
    lines.append(markdown_table(nontrusted, all_cols))

    lines.append("## Train Logs Without Aggregate Eval Rows\n")
    lines.append(
        markdown_table(
            train_only_rows,
            [
                ("status", "status"),
                ("run_id", "run_id"),
                ("trainer", "trainer"),
                ("backbone", "backbone"),
                ("shot", "shot"),
                ("seed", "seed"),
                ("last_epoch", "last_epoch_seen"),
                ("max_epoch", "max_epoch"),
                ("finished", "finished_training"),
            ],
        )
    )

    lines.append("## Metric CSV Inventory\n")
    inventory = {}
    for row in csv_rows:
        inventory[row["source_file"]] = inventory.get(row["source_file"], 0) + 1
    inventory_rows = [
        {"source_file": source_file, "rows": str(count)}
        for source_file, count in sorted(inventory.items())
    ]
    lines.append(markdown_table(inventory_rows, [("source_file", "source_file"), ("rows", "rows")]))

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not ROOT.exists():
        raise SystemExit("outputs directory does not exist")

    eval_rows = parse_eval_logs()
    eval_fields = [
        "source_file",
        "run_id",
        "eval_name",
        "shot",
        "seed",
        "trainer",
        "backbone",
        "load_epoch",
        "status",
        "accuracy",
        "top5_accuracy",
        "balanced_accuracy_present",
        "macro_f1_present",
        "macro_f1_all_182",
        "config_file",
    ]
    write_csv(ROOT / "all_eval_log_results.csv", eval_rows, eval_fields)

    train_only_rows = parse_train_logs_without_eval(eval_rows)
    write_csv(
        ROOT / "train_logs_without_eval.csv",
        train_only_rows,
        [
            "source_file",
            "run_id",
            "shot",
            "seed",
            "trainer",
            "backbone",
            "last_epoch_seen",
            "max_epoch",
            "finished_training",
            "status",
        ],
    )

    csv_rows, csv_fields = parse_metric_csvs()
    write_csv(ROOT / "all_metric_csv_rows.csv", csv_rows, csv_fields)
    write_markdown(ROOT / "experiment_results_all.md", eval_rows, csv_rows, train_only_rows)

    print(f"eval_log_rows={len(eval_rows)}")
    print(f"metric_csv_rows={len(csv_rows)}")
    print(f"train_logs_without_eval={len(train_only_rows)}")
    print("wrote outputs/all_eval_log_results.csv")
    print("wrote outputs/all_metric_csv_rows.csv")
    print("wrote outputs/train_logs_without_eval.csv")
    print("wrote outputs/experiment_results_all.md")


if __name__ == "__main__":
    main()
