import argparse
from pathlib import Path

import pandas as pd


BUCKETS = ("empty", "rare", "medium", "frequent")


def class_balanced_weight(counts, beta=0.9999):
    counts = counts.astype(float)
    safe = counts.clip(lower=1.0)
    weights = (1.0 - beta) / (1.0 - beta**safe)
    weights[counts <= 0] = 0.0
    positive = weights > 0
    weights[positive] = weights[positive] / weights[positive].mean()
    return weights


def frequency_bucket(y, classname, count, rare_threshold=20, medium_threshold=100):
    name = str(classname).lower().replace("_", " ")
    if int(y) == 0 or "empty" in name:
        return "empty"
    if count < rare_threshold:
        return "rare"
    if count < medium_threshold:
        return "medium"
    return "frequent"


def load_class_table(data_root):
    data_root = Path(data_root)
    meta = pd.read_csv(data_root / "metadata.csv")
    cats = pd.read_csv(data_root / "categories.csv")
    cats = cats[cats["y"].between(0, 181)].copy()
    cats["y"] = cats["y"].astype(int)

    train_counts = (
        meta[meta["split"].eq("train")]
        .groupby("y")
        .size()
        .rename("train_count")
        .reset_index()
    )
    train_counts["y"] = train_counts["y"].astype(int)
    table = cats.merge(train_counts, on="y", how="left").fillna({"train_count": 0})
    table["train_count"] = table["train_count"].astype(int)
    table["cb_weight"] = class_balanced_weight(table["train_count"])
    table["bucket"] = [
        frequency_bucket(row["y"], row["name"], row["train_count"])
        for _, row in table.iterrows()
    ]
    return table[["y", "name", "train_count", "cb_weight", "bucket"]]


def load_per_class(root, split):
    path = Path(root) / "shot_full" / f"eval_{split}" / "iwildcam_per_class_eval1.csv"
    df = pd.read_csv(path)
    return df.rename(
        columns={
            "precision": f"precision_{split}",
            "recall": f"recall_{split}",
            "f1": f"f1_{split}",
            "support": f"support_{split}",
        }
    )


def compare_split(wild_root, coop_root, class_table, split, out_dir):
    wild = load_per_class(wild_root, split)
    coop = load_per_class(coop_root, split)

    keep = ["y", "classname", f"support_{split}", f"precision_{split}", f"recall_{split}", f"f1_{split}"]
    merged = wild[keep].merge(
        coop[keep],
        on=["y", "classname", f"support_{split}"],
        suffixes=("_wild", "_coop"),
    )
    merged = merged.merge(class_table, on="y", how="left")
    merged["delta_f1"] = merged[f"f1_{split}_wild"] - merged[f"f1_{split}_coop"]
    merged["delta_precision"] = merged[f"precision_{split}_wild"] - merged[f"precision_{split}_coop"]
    merged["delta_recall"] = merged[f"recall_{split}_wild"] - merged[f"recall_{split}_coop"]
    merged["present"] = merged[f"support_{split}"] > 0

    present = merged[merged["present"]].copy()
    overall = {
        "split": split,
        "present_classes": int(present.shape[0]),
        "wild_macro_f1": present[f"f1_{split}_wild"].mean() * 100,
        "coop_macro_f1": present[f"f1_{split}_coop"].mean() * 100,
        "delta_macro_f1": present["delta_f1"].mean() * 100,
        "wild_macro_precision": present[f"precision_{split}_wild"].mean() * 100,
        "coop_macro_precision": present[f"precision_{split}_coop"].mean() * 100,
        "delta_macro_precision": present["delta_precision"].mean() * 100,
        "wild_macro_recall": present[f"recall_{split}_wild"].mean() * 100,
        "coop_macro_recall": present[f"recall_{split}_coop"].mean() * 100,
        "delta_macro_recall": present["delta_recall"].mean() * 100,
    }

    by_bucket = (
        present.groupby("bucket", dropna=False)
        .agg(
            classes=("y", "count"),
            eval_support=(f"support_{split}", "sum"),
            train_count_median=("train_count", "median"),
            cb_weight_median=("cb_weight", "median"),
            wild_f1=(f"f1_{split}_wild", "mean"),
            coop_f1=(f"f1_{split}_coop", "mean"),
            delta_f1=("delta_f1", "mean"),
            delta_precision=("delta_precision", "mean"),
            delta_recall=("delta_recall", "mean"),
        )
        .reset_index()
    )
    for col in ["wild_f1", "coop_f1", "delta_f1", "delta_precision", "delta_recall"]:
        by_bucket[col] *= 100

    drops = present.sort_values("delta_f1").head(20)
    gains = present.sort_values("delta_f1", ascending=False).head(20)
    merged.to_csv(out_dir / f"per_class_compare_{split}.csv", index=False)
    by_bucket.to_csv(out_dir / f"bucket_compare_{split}.csv", index=False)
    drops.to_csv(out_dir / f"top_f1_drops_{split}.csv", index=False)
    gains.to_csv(out_dir / f"top_f1_gains_{split}.csv", index=False)
    return overall, by_bucket, drops, gains


def load_train_history(run_dir):
    epoch_path = Path(run_dir) / "history" / "train_epoch_summary.csv"
    if not epoch_path.exists():
        return None
    return pd.read_csv(epoch_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wild-root", default="outputs/wildcoop_a_iwildcam_vitb16_full_seed1")
    parser.add_argument("--coop-root", default="outputs/coop_iwildcam_full_seed1")
    parser.add_argument("--data-root", default="data/iwildcam_v2.0")
    parser.add_argument("--out-dir", default="outputs/wildcoop_a_iwildcam_vitb16_full_seed1/diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_table = load_class_table(args.data_root)
    class_table.to_csv(out_dir / "class_counts_weights_buckets.csv", index=False)

    print("Class bucket/weight summary")
    bucket_summary = (
        class_table.groupby("bucket")
        .agg(
            classes=("y", "count"),
            train_count_min=("train_count", "min"),
            train_count_median=("train_count", "median"),
            train_count_max=("train_count", "max"),
            cb_weight_min=("cb_weight", "min"),
            cb_weight_median=("cb_weight", "median"),
            cb_weight_max=("cb_weight", "max"),
        )
        .reindex(BUCKETS)
        .reset_index()
    )
    print(bucket_summary.to_string(index=False))
    print()

    overalls = []
    split_details = {}
    for split in ["id_val", "id_test", "ood_val", "ood_test"]:
        overall, by_bucket, drops, gains = compare_split(args.wild_root, args.coop_root, class_table, split, out_dir)
        overalls.append(overall)
        split_details[split] = (by_bucket, drops, gains)

    overall_df = pd.DataFrame(overalls)
    overall_df.to_csv(out_dir / "overall_compare.csv", index=False)
    print("Overall WildCoOp-A vs CoOp")
    print(overall_df.round(2).to_string(index=False))
    print()

    for split in ["ood_val", "ood_test"]:
        by_bucket, drops, gains = split_details[split]
        print(f"Bucket comparison: {split}")
        print(by_bucket.round(2).to_string(index=False))
        print()
        cols = [
            "y",
            "classname",
            f"support_{split}",
            "bucket",
            "train_count",
            "cb_weight",
            f"f1_{split}_wild",
            f"f1_{split}_coop",
            "delta_f1",
            "delta_precision",
            "delta_recall",
        ]
        print(f"Top F1 drops: {split}")
        print(drops[cols].head(10).round(4).to_string(index=False))
        print()
        print(f"Top F1 gains: {split}")
        print(gains[cols].head(10).round(4).to_string(index=False))
        print()

    history = load_train_history(Path(args.wild_root) / "shot_full" / "train")
    if history is not None:
        best_acc = history.loc[history["acc_end_avg"].idxmax()]
        best_loss = history.loc[history["loss_end_avg"].idxmin()]
        final = history.iloc[-1]
        print("Train history diagnosis")
        print(
            f"final epoch={int(final.epoch)} loss={final.loss_end_avg:.5f} "
            f"acc={final.acc_end_avg:.2f} lr={final.lr_end:.2e}"
        )
        print(
            f"best train-acc epoch={int(best_acc.epoch)} acc={best_acc.acc_end_avg:.2f} "
            f"loss={best_acc.loss_end_avg:.5f}"
        )
        print(
            f"best loss epoch={int(best_loss.epoch)} loss={best_loss.loss_end_avg:.5f} "
            f"acc={best_loss.acc_end_avg:.2f}"
        )


if __name__ == "__main__":
    main()
