import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


ImageFile.LOAD_TRUNCATED_IMAGES = True


TEMPLATE_SETS = {
    "simple": [
        "a photo of {}.",
    ],
    "iwildcam": [
        "a photo of {}.",
        "{} in the wild.",
    ],
    "camera_trap": [
        "a camera trap photo of {}.",
        "a wildlife camera photo of {}.",
        "a photo of {} in the wild.",
    ],
}

EMPTY_PROMPT_SETS = {
    "none": [],
    "camera_trap": [
        "an empty camera trap photo.",
        "a camera trap photo with no animal.",
        "a wildlife camera photo with no animal.",
        "a photo of an empty scene.",
    ],
}


class IWildCamDataset(Dataset):
    def __init__(self, rows, image_dir, preprocess):
        self.rows = rows.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.image_dir / row["filename"]
        with Image.open(image_path) as image:
            image = self.preprocess(image)
        return image, int(row["y"]), row["filename"]


def convert_rgb(image):
    return image.convert("RGB")


def make_clip_preprocess(n_px):
    # Same normalization/resize as OpenAI CLIP, without a lambda so Windows workers can pickle it.
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            convert_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def repo_root():
    return Path(__file__).resolve().parent


def load_local_clip():
    coop_root = repo_root() / "external" / "CoOp"
    if not coop_root.exists():
        raise FileNotFoundError(f"Cannot find local CoOp repo at {coop_root}")
    sys.path.insert(0, str(coop_root))
    import clip  # noqa: WPS433

    return clip


def load_classnames(labels_csv, n_classes, empty_name=None):
    labels = pd.read_csv(labels_csv)
    labels = labels[labels["y"] < 99999].copy()
    labels["y"] = labels["y"].astype(int)
    labels = labels.sort_values("y")
    expected = list(range(n_classes))
    actual = labels["y"].tolist()
    if actual != expected:
        raise ValueError(f"Expected labels 0..{n_classes - 1}, got {actual[:5]}...{actual[-5:]}")

    classnames = labels["english"].astype(str).str.lower().tolist()
    if empty_name is not None:
        classnames[0] = empty_name
    return classnames


def build_text_features(clip, model, classnames, templates, device, empty_prompts=None):
    class_features = []
    with torch.no_grad():
        for class_idx, name in enumerate(classnames):
            if class_idx == 0 and empty_prompts:
                prompts = empty_prompts
            else:
                prompts = [template.format(name) for template in templates]
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            class_features.append(text_features)

    text_features = torch.cat(class_features, dim=0)

    return text_features


def sample_rows(rows, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    return rows.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def evaluate_split(args, split, rows, preprocess, model, text_features, device):
    dataset = IWildCamDataset(rows, Path(args.data_root) / "train", preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    y_true = []
    y_pred = []
    top5_hits = []
    filenames = []
    topk_size = min(5, text_features.shape[0])

    progress = tqdm(loader, desc=f"{split}", unit="batch")
    with torch.no_grad():
        for images, labels, batch_filenames in progress:
            images = images.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()
            topk = logits.topk(topk_size, dim=1).indices.cpu().numpy()
            labels_np = labels.numpy()

            y_true.extend(labels_np.tolist())
            y_pred.extend(topk[:, 0].tolist())
            top5_hits.extend((topk == labels_np[:, None]).any(axis=1).tolist())
            filenames.extend(batch_filenames)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    top5_hits = np.asarray(top5_hits, dtype=bool)
    labels_all = np.arange(args.n_classes)
    support = np.bincount(y_true, minlength=args.n_classes)
    present_labels = labels_all[support > 0]

    metrics = {
        "split": split,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top5_accuracy": float(top5_hits.mean()),
        "balanced_accuracy_present": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1_all_182": float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)),
        "macro_f1_present": float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)),
        "n_present_classes": int(len(present_labels)),
    }

    precision, recall, f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_all,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "split": split,
            "y": labels_all,
            "classname": args.classnames,
            "support": per_class_support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    predictions = None
    if args.save_predictions:
        predictions = pd.DataFrame(
            {
                "split": split,
                "filename": filenames,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )

    return metrics, per_class, predictions


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Zero-shot CLIP baseline for local iWildCam v2.0 data")
    parser.add_argument("--data-root", default=str(root / "data" / "iwildcam_v2.0"))
    parser.add_argument(
        "--labels-csv",
        default=str(root / "external" / "auto-ft" / "autoft" / "src" / "datasets" / "iwildcam_metadata" / "labels.csv"),
    )
    parser.add_argument("--output-dir", default=str(root / "outputs" / "zero_shot_iwildcam"))
    parser.add_argument("--model", default="ViT-B/16", choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"])
    parser.add_argument("--template-set", default="iwildcam", choices=sorted(TEMPLATE_SETS))
    parser.add_argument("--empty-prompt-set", default="none", choices=sorted(EMPTY_PROMPT_SETS))
    parser.add_argument("--splits", nargs="+", default=["id_val", "id_test", "val", "test"])
    parser.add_argument("--n-classes", type=int, default=182)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--empty-name", default=None, help="Optional replacement classname for y=0")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    metadata_csv = data_root / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Cannot find metadata.csv at {metadata_csv}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.classnames = load_classnames(args.labels_csv, args.n_classes, args.empty_name)
    templates = TEMPLATE_SETS[args.template_set]
    empty_prompts = EMPTY_PROMPT_SETS[args.empty_prompt_set]

    clip = load_local_clip()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Loading CLIP model: {args.model}")
    model, _ = clip.load(args.model, device=device, jit=False)
    model.eval()
    preprocess = make_clip_preprocess(model.visual.input_resolution)

    print(f"Templates ({args.template_set}): {templates}")
    if empty_prompts:
        print(f"Empty prompts ({args.empty_prompt_set}): {empty_prompts}")
    text_features = build_text_features(clip, model, args.classnames, templates, device, empty_prompts)

    metadata = pd.read_csv(metadata_csv)
    metadata["y"] = metadata["y"].astype(int)
    all_metrics = []
    per_class_parts = []
    prediction_parts = []

    for split in args.splits:
        rows = metadata[metadata["split"] == split].copy()
        if rows.empty:
            raise ValueError(f"Split {split!r} has no rows in {metadata_csv}")
        rows = sample_rows(rows, args.max_samples_per_split, args.seed)
        print(f"Evaluating split={split}, n={len(rows)}")
        metrics, per_class, predictions = evaluate_split(args, split, rows, preprocess, model, text_features, device)
        all_metrics.append(metrics)
        per_class_parts.append(per_class)
        if predictions is not None:
            prediction_parts.append(predictions)
        print(json.dumps(metrics, indent=2))

    model_slug = args.model.replace("/", "-")
    run_name = args.run_name or f"{model_slug}_{args.template_set}"
    if args.max_samples_per_split > 0:
        run_name = f"{run_name}_sample{args.max_samples_per_split}"

    summary = pd.DataFrame(all_metrics)
    summary_path = output_dir / f"{run_name}_metrics.csv"
    per_class_path = output_dir / f"{run_name}_per_class.csv"
    json_path = output_dir / f"{run_name}_metrics.json"
    summary.to_csv(summary_path, index=False)
    pd.concat(per_class_parts, ignore_index=True).to_csv(per_class_path, index=False)

    payload = {
        "model": args.model,
        "template_set": args.template_set,
        "templates": templates,
        "empty_prompt_set": args.empty_prompt_set,
        "empty_prompts": empty_prompts,
        "splits": args.splits,
        "n_classes": args.n_classes,
        "empty_name": args.empty_name,
        "max_samples_per_split": args.max_samples_per_split,
        "metrics": all_metrics,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    if prediction_parts:
        pred_path = output_dir / f"{run_name}_predictions.csv"
        pd.concat(prediction_parts, ignore_index=True).to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

    print(f"Saved metrics: {summary_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved per-class metrics: {per_class_path}")


if __name__ == "__main__":
    # Multiprocessing DataLoader on Windows needs this guard.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
