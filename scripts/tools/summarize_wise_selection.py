import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "accuracy",
    "top5_accuracy",
    "balanced_accuracy_present",
    "macro_f1_present",
    "macro_f1_all_182",
]


def main():
    parser = argparse.ArgumentParser(description="Summarize per-seed selected WiSE variants from multiseed detail CSV.")
    parser.add_argument("--detail", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate-output", required=True)
    parser.add_argument("--selection-name", required=True)
    parser.add_argument("--selection-split", required=True)
    parser.add_argument("--selection-metric", required=True)
    parser.add_argument("--shots", type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.detail)
    if df.empty:
        raise ValueError(f"No rows found in {args.detail}")
    if args.selection_metric not in df.columns:
        raise ValueError(f"Metric {args.selection_metric!r} not found in {args.detail}")

    rows = []
    for seed in sorted(df["seed"].unique()):
        seed_df = df[(df["seed"] == seed) & (df["shots"] == args.shots)]
        candidates = seed_df[
            (seed_df["split"] == args.selection_split)
            & seed_df["variant"].astype(str).str.startswith("a")
        ].copy()
        if candidates.empty:
            continue

        selected = candidates.sort_values(args.selection_metric, ascending=False).iloc[0]
        variant = selected["variant"]
        row = {
            "selection": args.selection_name,
            "shots": args.shots,
            "seed": int(seed),
            "variant": variant,
            "selection_split": args.selection_split,
            "selection_metric": args.selection_metric,
            "selection_value": selected[args.selection_metric],
        }

        for split in ["id_val", "ood_val", "id_test", "ood_test"]:
            split_rows = seed_df[(seed_df["split"] == split) & (seed_df["variant"] == variant)]
            if split_rows.empty:
                continue
            split_row = split_rows.iloc[0]
            for metric in METRICS:
                if metric in split_row:
                    row[f"{split}_{metric}"] = split_row[metric]

        rows.append(row)

    detail = pd.DataFrame(rows)
    metric_cols = [col for col in detail.columns if col not in {"selection", "shots", "seed", "variant", "selection_split", "selection_metric"}]
    aggregate = detail.groupby(["selection", "shots"], as_index=False)[metric_cols].agg(["count", "mean", "std"])
    aggregate.columns = ["_".join(col).rstrip("_") for col in aggregate.columns.to_flat_index()]

    output = Path(args.output)
    aggregate_output = Path(args.aggregate_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)
    aggregate.to_csv(aggregate_output, index=False)

    print("Selected results:")
    print(detail.to_string(index=False))
    print(f"Saved selected details: {output}")
    print("\nSelected aggregate:")
    print(aggregate.to_string(index=False))
    print(f"Saved selected aggregate: {aggregate_output}")


if __name__ == "__main__":
    main()
