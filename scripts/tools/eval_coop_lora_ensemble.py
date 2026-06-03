import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm


def add_repo_paths(coop_root, dassl_root):
    for path in (str(dassl_root), str(coop_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def setup_cfg(args):
    from dassl.config import get_cfg_default
    from train import extend_cfg

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
    from trainers.coop_lora import CoOpLoRATrainable, CustomCLIPCoOpLoRA, inject_text_lora, inject_visual_lora
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
    incompatible = trainable.load_state_dict(checkpoint["state_dict"], strict=False)
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


def parse_model_spec(spec):
    if "=" not in spec:
        return Path(spec), 1
    key, value = spec.split("=", 1)
    if key != "path":
        raise ValueError(f"Unsupported model spec {spec!r}; expected path=<path> or <path>")
    return Path(value), 1


def evaluate(args, cfg, data_loader, lab2cname, classnames):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
    model_specs = [parse_model_spec(spec) for spec in args.model]
    models = [build_model(cfg, classnames, model_path, epoch, device) for model_path, epoch in model_specs]
    print(f"Evaluating logit ensemble with {len(models)} models")

    y_true = []
    y_pred = []
    top5_correct = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            image = batch["img"].to(device)
            label = batch["label"].to(device)
            logits_sum = None
            for model in models:
                logits = model(image).float()
                logits_sum = logits if logits_sum is None else logits_sum + logits
            logits = logits_sum / float(len(models))

            pred = logits.argmax(dim=1)
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
            top5_correct += int(top5.eq(label.view(-1, 1)).any(dim=1).sum().item())
            y_true.extend(label.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    metrics, per_class = compute_metrics(y_true, y_pred, top5_correct, lab2cname, len(classnames))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = OrderedDict()
    row["split"] = args.split_name
    row["ensemble"] = args.ensemble_name
    row["num_models"] = len(models)
    row.update(metrics)
    summary_path = output_dir / "summary.csv"
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    per_class.to_csv(output_dir / f"per_class_{args.ensemble_name}_{args.split_name}.csv", index=False)

    print(
        "=> iWildCam ensemble result\n"
        f"* split: {args.split_name}\n"
        f"* ensemble: {args.ensemble_name}\n"
        f"* total: {len(y_true):,}\n"
        f"* accuracy: {metrics['accuracy']:.2f}%\n"
        f"* top5_accuracy: {metrics['top5_accuracy']:.2f}%\n"
        f"* balanced_accuracy_present: {metrics['balanced_accuracy_present']:.2f}%\n"
        f"* macro_f1_present: {metrics['macro_f1_present']:.2f}%\n"
        f"* macro_f1_all_182: {metrics['macro_f1_all_182']:.2f}%"
    )
    print(f"Saved summary: {summary_path}")
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CoOpLoRA logit ensemble.")
    parser.add_argument("--repo-root", default="D:/Workplace/DG-iWildCam")
    parser.add_argument("--coop-root", default=None)
    parser.add_argument("--dassl-root", default=None)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-config-file", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ensemble-name", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--test-split", required=True, choices=["val", "test"])
    parser.add_argument("--iwildcam-test-split", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--num-shots", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
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
    evaluate(args, cfg, data_loader, dm.lab2cname, dm.dataset.classnames)


if __name__ == "__main__":
    main()
