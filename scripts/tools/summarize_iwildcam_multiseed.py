import argparse
import re
from pathlib import Path

import pandas as pd

from summarize_iwildcam import parse_eval_dir


SPLITS = ("ood_test", "ood_val", "id_test", "id_val")


def parse_eval_name(eval_dir):
    name = eval_dir.name
    if not name.startswith("eval_"):
        return None, None

    token = name.replace("eval_", "", 1)
    for split in SPLITS:
        if token == split:
            return split, "default"
        prefix = split + "_"
        if token.startswith(prefix):
            return split, token.replace(prefix, "", 1)

    return token, "default"


def parse_shots(shot_dir):
    token = shot_dir.name.replace("shot_", "", 1)
    if token in {"full", "all"}:
        return -1
    return int(token)


def parse_seed(seed_dir):
    match = re.fullmatch(r"seed_(\d+)", seed_dir.name)
    if match is None:
        raise ValueError(f"Could not parse seed from {seed_dir}")
    return int(match.group(1))


def summarize(root):
    rows = []
    for eval_dir in sorted(root.glob("seed_*/shot_*/eval_*")):
        metrics = parse_eval_dir(eval_dir)
        if metrics is None:
            continue

        split, variant = parse_eval_name(eval_dir)
        rows.append(
            {
                "shots": parse_shots(eval_dir.parent),
                "seed": parse_seed(eval_dir.parent.parent),
                "variant": variant,
                "split": split,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["shots", "variant", "seed", "split"])
    return df


def aggregate(df):
    if df.empty:
        return df

    metric_cols = [
        col
        for col in df.columns
        if col not in {"shots", "seed", "variant", "split"}
    ]
    agg = df.groupby(["shots", "variant", "split"], as_index=False)[metric_cols].agg(["count", "mean", "std"])
    agg.columns = ["_".join(col).rstrip("_") for col in agg.columns.to_flat_index()]
    return agg.sort_values(["shots", "variant", "split"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate-output", required=True)
    args = parser.parse_args()

    detail = summarize(Path(args.root))
    aggregate_df = aggregate(detail)

    output = Path(args.output)
    aggregate_output = Path(args.aggregate_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output, index=False)
    aggregate_df.to_csv(aggregate_output, index=False)

    print("Detailed results:")
    print(detail.to_string(index=False))
    print(f"Saved details: {output}")
    print("\nAggregate results:")
    print(aggregate_df.to_string(index=False))
    print(f"Saved aggregate: {aggregate_output}")


if __name__ == "__main__":
    main()
