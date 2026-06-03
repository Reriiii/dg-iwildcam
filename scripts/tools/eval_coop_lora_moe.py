import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm


BUCKETS = ("empty", "rare", "medium", "frequent")


def add_repo_paths(coop_root, dassl_root):
    for path in (str(dassl_root), str(coop_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def setup_cfg(args):
    from dassl.config import get_cfg_default
    from train import extend_cfg  # noqa: F401; also registers CoOp datasets/trainers

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_file(args.config_file)

    cfg.DATASET.ROOT = args.root
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.SEED = args.seed
    cfg.DATASET.NUM_SHOTS = args.num_shots
    cfg.TEST.SPLIT = args.test_split
    cfg.DATASET.IWILDCAM_TEST_SPLIT = args.iwildcam_test_split
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.VERBOSE = args.verbose
    cfg.freeze()
    return cfg


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint(model_dir_or_file, epoch):
    path = Path(model_dir_or_file)
    if path.is_file():
        return path
    return path / "coop_lora" / f"model.pth.tar-{epoch}"


def build_model(cfg, classnames, model_dir_or_file, epoch, device):
    from trainers.coop_lora import (
        CoOpLoRATrainable,
        CustomCLIPCoOpLoRA,
        inject_text_lora,
        inject_visual_lora,
    )
    from trainers.coop import load_clip_to_cpu

    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
        clip_model.float()

    visual_lora_modules = inject_visual_lora(clip_model, cfg.TRAINER.COOPLORA)
    text_lora_modules = inject_text_lora(clip_model, cfg.TRAINER.COOPLORA)
    lora_modules = torch.nn.ModuleList([*visual_lora_modules, *text_lora_modules])
    model = CustomCLIPCoOpLoRA(cfg, classnames, clip_model)
    trainable = CoOpLoRATrainable(model, lora_modules)

    checkpoint_path = resolve_checkpoint(model_dir_or_file, epoch)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = checkpoint["state_dict"]
    incompatible = trainable.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys for {checkpoint_path}: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys for {checkpoint_path}: {incompatible.unexpected_keys}")

    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    model.eval()
    print(f"Loaded CoOpLoRA model: {checkpoint_path}")
    return model


def class_counts_from_metadata(cfg, num_classes):
    root = Path(cfg.DATASET.ROOT).expanduser().resolve()
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        metadata_path = root / "iwildcam_v2.0" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found under {root}")

    metadata = pd.read_csv(metadata_path)
    metadata["y"] = metadata["y"].astype(int)
    rows = metadata[(metadata["split"] == "train") & (metadata["y"] < num_classes)]
    counts = np.bincount(rows["y"].to_numpy(dtype=np.int64), minlength=num_classes)
    return counts


def build_bucket_ids(cfg, classnames, rare_threshold, medium_threshold):
    counts = class_counts_from_metadata(cfg, len(classnames))
    bucket_ids = np.zeros(len(classnames), dtype=np.int64)
    for label, (name, count) in enumerate(zip(classnames, counts.tolist())):
        normalized = str(name).lower().replace("_", " ")
        if label == 0 or "empty" in normalized:
            bucket = "empty"
        elif count < rare_threshold:
            bucket = "rare"
        elif count < medium_threshold:
            bucket = "medium"
        else:
            bucket = "frequent"
        bucket_ids[label] = BUCKETS.index(bucket)
    bucket_counts = {bucket: int((bucket_ids == idx).sum()) for idx, bucket in enumerate(BUCKETS)}
    print(f"Frequency buckets from full train counts: {bucket_counts}")
    return bucket_ids, counts


def format_float(value):
    return (f"{value:.2f}" if value < 1 else f"{value:.1f}").rstrip("0").rstrip(".").replace(".", "p")


def add_candidate(candidates, name, empty, rare, medium, frequent):
    values = {
        "empty": float(empty),
        "rare": float(rare),
        "medium": float(medium),
        "frequent": float(frequent),
    }
    for bucket, value in values.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} has invalid {bucket} lambda={value}")
    candidates[name] = values


def parse_candidate_spec(spec):
    if ":" not in spec:
        raise ValueError(f"Candidate must be name:key=value,... got {spec!r}")
    name, rest = spec.split(":", 1)
    values = {}
    for item in rest.split(","):
        key, value = item.split("=", 1)
        values[key.strip()] = float(value)
    missing = set(BUCKETS) - set(values)
    if missing:
        raise ValueError(f"Candidate {name!r} missing bucket lambdas: {sorted(missing)}")
    return name, values


def build_candidates(args):
    candidates = OrderedDict()
    for spec in args.candidate:
        name, values = parse_candidate_spec(spec)
        add_candidate(candidates, name, values["empty"], values["rare"], values["medium"], values["frequent"])

    grids = set(args.grid)
    if "global" in grids:
        for value in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 1.0]:
            token = format_float(value)
            add_candidate(candidates, f"g{token}", value, value, value, value)

    if "bucket" in grids:
        empty_values = [0.0]
        frequent_values = [0.0, 0.03, 0.05, 0.10]
        medium_values = [0.05, 0.10, 0.20, 0.30]
        rare_values = [0.10, 0.20, 0.40, 0.60, 0.80]
        for empty in empty_values:
            for frequent in frequent_values:
                for medium in medium_values:
                    for rare in rare_values:
                        if not (rare >= medium >= frequent):
                            continue
                        name = (
                            f"b_e{format_float(empty)}"
                            f"_f{format_float(frequent)}"
                            f"_m{format_float(medium)}"
                            f"_r{format_float(rare)}"
                        )
                        add_candidate(candidates, name, empty, rare, medium, frequent)

    if args.only_candidate_name:
        keep = set(args.only_candidate_name)
        candidates = OrderedDict((name, values) for name, values in candidates.items() if name in keep)
        missing = keep - set(candidates)
        if missing:
            raise ValueError(f"Requested candidate names not found in generated grid: {sorted(missing)}")

    if not candidates:
        raise ValueError("No MoE candidates requested")
    print(f"Evaluating {len(candidates)} MoE candidates")
    return candidates


def lambda_vector_for_candidate(candidate, bucket_ids, device):
    values = np.zeros(len(bucket_ids), dtype=np.float32)
    for bucket_idx, bucket in enumerate(BUCKETS):
        values[bucket_ids == bucket_idx] = candidate[bucket]
    return torch.from_numpy(values).to(device).view(1, -1)


def compute_metrics(y_true, y_pred, top5_correct, lab2cname, num_classes):
    labels_all = np.arange(num_classes)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    support = np.bincount(y_true, minlength=num_classes)
    present_labels = labels_all[support > 0]
    precision, recall, f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )
    metrics = OrderedDict()
    metrics["accuracy"] = 100.0 * accuracy_score(y_true, y_pred)
    metrics["top5_accuracy"] = 100.0 * float(top5_correct) / float(len(y_true))
    metrics["balanced_accuracy_present"] = 100.0 * float(np.mean(recall[present_labels]))
    metrics["macro_f1_present"] = 100.0 * f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)
    metrics["macro_f1_all_182"] = 100.0 * f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)
    per_class = pd.DataFrame(
        {
            "y": labels_all,
            "classname": [lab2cname.get(int(label), str(label)) for label in labels_all],
            "support": per_class_support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return metrics, per_class


def evaluate(args, cfg, data_loader, lab2cname, classnames, bucket_ids, candidates):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
    generalist = build_model(cfg, classnames, args.generalist_model, args.generalist_epoch, device)
    specialist = build_model(cfg, classnames, args.specialist_model, args.specialist_epoch, device)

    lambda_vectors = {
        name: lambda_vector_for_candidate(values, bucket_ids, device)
        for name, values in candidates.items()
    }
    y_true = []
    y_pred = {name: [] for name in candidates}
    top5_correct = {name: 0 for name in candidates}

    with torch.no_grad():
        for batch in tqdm(data_loader):
            image = batch["img"].to(device)
            label = batch["label"].to(device)
            logits_generalist = generalist(image).float() / args.generalist_temperature
            logits_specialist = specialist(image).float() / args.specialist_temperature
            y_true.extend(label.cpu().numpy().tolist())

            for name, lambda_vec in lambda_vectors.items():
                logits = logits_generalist * (1.0 - lambda_vec) + logits_specialist * lambda_vec
                pred = logits.argmax(dim=1)
                top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
                top5_correct[name] += int(top5.eq(label.view(-1, 1)).any(dim=1).sum().item())
                y_pred[name].extend(pred.cpu().numpy().tolist())

    rows = []
    per_class_outputs = {}
    for name, values in candidates.items():
        metrics, per_class = compute_metrics(y_true, y_pred[name], top5_correct[name], lab2cname, len(classnames))
        row = OrderedDict()
        row["split"] = args.split_name
        row["candidate"] = name
        for bucket in BUCKETS:
            row[f"lambda_{bucket}"] = values[bucket]
        row.update(metrics)
        rows.append(row)
        per_class_outputs[name] = per_class
        print(
            f"{args.split_name} {name}: "
            f"acc={metrics['accuracy']:.2f} "
            f"bal={metrics['balanced_accuracy_present']:.2f} "
            f"macro_f1={metrics['macro_f1_present']:.2f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if args.write_per_class:
        for name, per_class in per_class_outputs.items():
            per_class.to_csv(output_dir / f"per_class_{name}.csv", index=False)
    print(f"Saved summary: {summary_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate frequency-routed CoOpLoRA logit MoE.")
    parser.add_argument("--repo-root", default="D:/Workplace/DG-iWildCam")
    parser.add_argument("--coop-root", default=None)
    parser.add_argument("--dassl-root", default=None)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-config-file", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generalist-model", required=True)
    parser.add_argument("--specialist-model", required=True)
    parser.add_argument("--generalist-epoch", type=int, default=1)
    parser.add_argument("--specialist-epoch", type=int, default=1)
    parser.add_argument("--generalist-temperature", type=float, default=1.0)
    parser.add_argument("--specialist-temperature", type=float, default=1.0)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--test-split", required=True, choices=["val", "test"])
    parser.add_argument("--iwildcam-test-split", required=True)
    parser.add_argument("--num-shots", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rare-threshold", type=int, default=20)
    parser.add_argument("--medium-threshold", type=int, default=100)
    parser.add_argument("--grid", action="append", default=[], choices=["global", "bucket"])
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--only-candidate-name", action="append", default=[])
    parser.add_argument("--write-per-class", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    coop_root = Path(args.coop_root) if args.coop_root else repo_root / "external" / "CoOp"
    dassl_root = Path(args.dassl_root) if args.dassl_root else repo_root / "external" / "Dassl.pytorch"
    args.output_dir = Path(args.output_dir)
    add_repo_paths(coop_root, dassl_root)

    from dassl.data import DataManager

    cfg = setup_cfg(args)
    dm = DataManager(cfg)
    data_loader = dm.val_loader if args.test_split == "val" else dm.test_loader
    bucket_ids, _ = build_bucket_ids(cfg, dm.dataset.classnames, args.rare_threshold, args.medium_threshold)
    candidates = build_candidates(args)
    evaluate(args, cfg, data_loader, dm.lab2cname, dm.dataset.classnames, bucket_ids, candidates)


if __name__ == "__main__":
    main()
