import argparse
import re
from pathlib import Path

import pandas as pd


METRIC_PATTERNS = {
    "accuracy": re.compile(r"\* accuracy: ([0-9.]+)%"),
    "top5_accuracy": re.compile(r"\* top5_accuracy: ([0-9.]+)%"),
    "balanced_accuracy_present": re.compile(r"\* balanced_accuracy_present: ([0-9.]+)%"),
    "macro_f1_present": re.compile(r"\* macro_f1_present: ([0-9.]+)%"),
    "macro_f1_all_182": re.compile(r"\* macro_f1_all_182: ([0-9.]+)%"),
}


def read_latest_log(directory):
    logs = sorted(Path(directory).glob("log.txt*"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    return logs[-1].read_text(encoding="utf-8", errors="ignore")


def parse_eval_dir(eval_dir):
    text = read_latest_log(eval_dir)
    if text is None or "=> iWildCam result" not in text:
        return None

    row = {}
    for key, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            row[key] = float(matches[-1])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/coop_iwildcam_seed1")
    parser.add_argument("--output", default="outputs/coop_iwildcam_seed1_summary.csv")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    seed = args.seed
    if seed is None:
        match = re.search(r"seed(\d+)", str(root))
        seed = int(match.group(1)) if match else 1

    rows = []
    for shot_dir in sorted(root.glob("shot_*")):
        try:
            shot_token = shot_dir.name.split("_")[1]
            shots = -1 if shot_token in {"full", "all"} else int(shot_token)
        except (IndexError, ValueError):
            continue

        for eval_dir in sorted(shot_dir.glob("eval_*")):
            metrics = parse_eval_dir(eval_dir)
            if metrics is None:
                continue
            split = eval_dir.name.replace("eval_", "")
            rows.append({"shots": shots, "seed": seed, "split": split, **metrics})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["shots", "split"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(df.to_string(index=False))
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()
