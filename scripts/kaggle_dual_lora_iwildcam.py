import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm.auto import tqdm

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


ImageFile.LOAD_TRUNCATED_IMAGES = True
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
WILDS_SPLIT_MAP = {0: "train", 1: "val", 2: "test", 3: "id_val", 4: "id_test"}


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1_present: float
    macro_f1_all: float
    balanced_accuracy_present: float
    n: int
    n_present_classes: int


class ConvertRGB:
    def __call__(self, image):
        return image.convert("RGB")


class IWildCamCsvDataset(Dataset):
    def __init__(self, rows, dataset_root, image_root, transform, filename_col):
        self.rows = rows.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.image_root = Path(image_root)
        self.transform = transform
        self.filename_col = filename_col

    def __len__(self):
        return len(self.rows)

    def _resolve_image_path(self, filename):
        path = Path(str(filename))
        if path.is_absolute() and path.exists():
            return path

        candidates = [
            self.image_root / path,
            self.dataset_root / path,
            self.dataset_root / "train" / path,
            self.dataset_root / "images" / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Return the primary candidate so the FileNotFoundError contains a useful path.
        return candidates[0]

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self._resolve_image_path(row[self.filename_col])
        with Image.open(image_path) as image:
            pixel_values = self.transform(image)
        label = int(row["y"])
        return pixel_values, label


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.runtime_scale = 1.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        result = self.base(x)
        update = F.linear(self.dropout(x), self.lora_A)
        update = F.linear(update, self.lora_B)
        return result + update * self.scaling * self.runtime_scale


class DualLoRACLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        class_input_ids,
        class_attention_mask,
        vision_layers,
        text_layers,
        vision_rank,
        text_rank,
        vision_alpha,
        text_alpha,
        lora_dropout,
        targets,
    ):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad_(False)

        self.register_buffer("class_input_ids", class_input_ids, persistent=False)
        self.register_buffer("class_attention_mask", class_attention_mask, persistent=False)
        self.targets = tuple(targets)
        self.injected_modules = self._inject_dual_lora(
            vision_layers=vision_layers,
            text_layers=text_layers,
            vision_rank=vision_rank,
            text_rank=text_rank,
            vision_alpha=vision_alpha,
            text_alpha=text_alpha,
            lora_dropout=lora_dropout,
        )

    def _replace_attention_projection(self, attention, attr_name, rank, alpha, dropout):
        if not hasattr(attention, attr_name):
            raise AttributeError(f"Attention module {attention.__class__.__name__} has no {attr_name}")
        base = getattr(attention, attr_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected {attr_name} to be nn.Linear, got {type(base)!r}")
        setattr(attention, attr_name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        return 1

    def _inject_side(self, layers, n_last_layers, rank, alpha, dropout):
        if n_last_layers <= 0:
            return 0
        selected_layers = layers[-min(n_last_layers, len(layers)) :]
        count = 0
        for layer in selected_layers:
            attention = layer.self_attn
            for target in self.targets:
                attr_name = target if target.endswith("_proj") else f"{target}_proj"
                count += self._replace_attention_projection(attention, attr_name, rank, alpha, dropout)
        return count

    def _inject_dual_lora(self, vision_layers, text_layers, vision_rank, text_rank, vision_alpha, text_alpha, lora_dropout):
        vision_count = self._inject_side(
            self.clip.vision_model.encoder.layers,
            vision_layers,
            vision_rank,
            vision_alpha,
            lora_dropout,
        )
        text_count = self._inject_side(
            self.clip.text_model.encoder.layers,
            text_layers,
            text_rank,
            text_alpha,
            lora_dropout,
        )
        return {"vision": vision_count, "text": text_count}

    def forward(self, pixel_values):
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        text_features = self.clip.get_text_features(
            input_ids=self.class_input_ids,
            attention_mask=self.class_attention_mask,
        )
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        logit_scale = self.clip.logit_scale.exp().float()
        return logit_scale * image_features @ text_features.t()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kaggle iWildCam baseline: CLIP ViT-B/16 dual LoRA with CE, ID/OOD macro-F1, WiSE-FT, wandb, tqdm."
    )
    parser.add_argument("--data-root", required=True, help="Path containing metadata.csv or an iwildcam_v2.0 folder")
    parser.add_argument("--image-root", default=None, help="Optional image folder; defaults to <dataset-root>/train")
    parser.add_argument("--categories-csv", default=None, help="Optional categories/labels CSV with y and class name columns")
    parser.add_argument("--output-dir", default="/kaggle/working/dual_lora_iwildcam")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch16")
    parser.add_argument("--prompt-template", default="a camera trap photo of {}.")
    parser.add_argument("--empty-class-name", default=None, help="Optional replacement text for label 0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128, help="Total batch size. DataParallel splits it across GPUs.")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--sampler", choices=["random", "class_balanced"], default="class_balanced")
    parser.add_argument("--class-balanced-power", type=float, default=0.5)
    parser.add_argument("--vision-layers", type=int, default=8)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--vision-rank", type=int, default=4)
    parser.add_argument("--text-rank", type=int, default=2)
    parser.add_argument("--vision-alpha", type=float, default=8.0)
    parser.add_argument("--text-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", nargs="+", default=["q", "v"], choices=["q", "k", "v", "out", "q_proj", "k_proj", "v_proj", "out_proj"])
    parser.add_argument("--wise-alphas", default="0.0,0.3,0.5,0.6,0.7,0.75,0.8,0.85,0.9,1.0")
    parser.add_argument("--wise-select", choices=["ood_val", "id_val"], default="ood_val")
    parser.add_argument("--wise-ensemble-top-k", type=int, default=3)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-install", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="dg-iwildcam-dual-lora")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def ensure_optional_dependencies(auto_install, wandb_mode):
    packages = [
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
    ]
    if wandb_mode != "disabled":
        packages.append(("wandb", "wandb"))
    missing = []
    for module_name, package_name in packages:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing and not auto_install:
        raise ImportError(f"Missing packages: {missing}. Re-run with --auto-install or install them in the Kaggle notebook.")
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def find_dataset_root(data_root):
    data_root = Path(data_root)
    candidates = [data_root, data_root / "iwildcam_v2.0"]
    if data_root.exists():
        candidates.extend(path for path in data_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "metadata.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not find metadata.csv under {data_root}")


def find_filename_column(metadata):
    for column in ["filename", "file_name", "filepath", "path", "image_path"]:
        if column in metadata.columns:
            return column
    raise ValueError(
        "Could not infer image filename column. Expected one of: filename, file_name, filepath, path, image_path."
    )


def normalize_splits(metadata):
    metadata = metadata.copy()
    if "split" in metadata.columns:
        split_values = metadata["split"]
        numeric_splits = pd.to_numeric(split_values, errors="coerce")
        if numeric_splits.notna().all() and set(numeric_splits.astype(int).unique()).issubset(WILDS_SPLIT_MAP):
            metadata["split"] = numeric_splits.astype(int).map(WILDS_SPLIT_MAP)
        else:
            metadata["split"] = split_values.astype(str)
        return metadata
    if "split_id" in metadata.columns:
        metadata["split"] = metadata["split_id"].astype(int).map(WILDS_SPLIT_MAP)
        if metadata["split"].isna().any():
            bad_values = sorted(metadata.loc[metadata["split"].isna(), "split_id"].unique().tolist())
            raise ValueError(f"Unknown split_id values: {bad_values}")
        return metadata
    raise ValueError("metadata.csv must contain either split or split_id")


def load_metadata(dataset_root):
    metadata = pd.read_csv(Path(dataset_root) / "metadata.csv")
    if "y" not in metadata.columns:
        raise ValueError("metadata.csv must contain a y label column")
    metadata = normalize_splits(metadata)
    metadata["y"] = metadata["y"].astype(int)
    labeled = metadata[metadata["y"].between(0, 99998)].copy()
    if len(labeled) != len(metadata):
        print(f"Dropping {len(metadata) - len(labeled)} rows with unsupported labels outside [0, 99998].", flush=True)
    metadata = labeled
    return metadata


def resolve_categories_csv(dataset_root, categories_csv):
    if categories_csv:
        path = Path(categories_csv)
        if path.exists():
            return path
        raise FileNotFoundError(path)
    for name in ["categories.csv", "labels.csv"]:
        path = Path(dataset_root) / name
        if path.exists():
            return path
    return None


def load_classnames(dataset_root, categories_csv, num_classes, empty_class_name):
    path = resolve_categories_csv(dataset_root, categories_csv)
    classnames = [f"class {idx}" for idx in range(num_classes)]
    if path is not None:
        table = pd.read_csv(path)
        if "y" not in table.columns:
            print(f"Warning: {path} has no y column; using generic class names.")
        else:
            name_col = next((col for col in ["english", "name", "category_name", "class_name"] if col in table.columns), None)
            if name_col is None:
                print(f"Warning: {path} has no supported class name column; using generic class names.")
            else:
                table = table[table["y"].between(0, num_classes - 1)].copy()
                for _, row in table.iterrows():
                    classnames[int(row["y"])] = str(row[name_col]).replace("_", " ").lower()

    if empty_class_name is not None and num_classes > 0:
        classnames[0] = empty_class_name
    return classnames


def make_transform(image_size):
    return Compose(
        [
            Resize(image_size, interpolation=BICUBIC),
            CenterCrop(image_size),
            ConvertRGB(),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def maybe_sample_rows(rows, max_samples, seed):
    if max_samples is None or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    return rows.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def make_split_rows(metadata, split_name, max_samples, seed):
    rows = metadata[metadata["split"].eq(split_name)].copy()
    if rows.empty:
        raise ValueError(f"Split {split_name!r} has no rows")
    return maybe_sample_rows(rows, max_samples=max_samples, seed=seed)


def make_sampler(rows, num_classes, power):
    counts = np.bincount(rows["y"].to_numpy(dtype=np.int64), minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights_per_class = np.power(safe_counts, -power)
    sample_weights = weights_per_class[rows["y"].to_numpy(dtype=np.int64)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(rows),
        replacement=True,
    )


def build_loaders(args, metadata, dataset_root, image_root, filename_col, num_classes):
    transform = make_transform(args.image_size)
    train_rows = make_split_rows(metadata, "train", args.max_train_samples, args.seed)
    eval_max = args.max_eval_samples
    split_rows = {
        "id_val": make_split_rows(metadata, "id_val", eval_max, args.seed),
        "ood_val": make_split_rows(metadata, "val", eval_max, args.seed),
        "id_test": make_split_rows(metadata, "id_test", eval_max, args.seed),
        "ood_test": make_split_rows(metadata, "test", eval_max, args.seed),
    }

    train_dataset = IWildCamCsvDataset(train_rows, dataset_root, image_root, transform, filename_col)
    eval_datasets = {
        name: IWildCamCsvDataset(rows, dataset_root, image_root, transform, filename_col)
        for name, rows in split_rows.items()
    }

    sampler = None
    shuffle = True
    if args.sampler == "class_balanced":
        sampler = make_sampler(train_rows, num_classes, args.class_balanced_power)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        for name, dataset in eval_datasets.items()
    }
    return train_loader, eval_loaders, split_rows


def build_class_tokens(tokenizer, classnames, prompt_template):
    prompts = [prompt_template.format(name) for name in classnames]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"], prompts


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def set_lora_scale(model, scale):
    raw_model = unwrap_model(model)
    for module in raw_model.modules():
        if isinstance(module, LoRALinear):
            module.runtime_scale = float(scale)


def trainable_state_dict(model):
    raw_model = unwrap_model(model)
    trainable_names = {name for name, param in raw_model.named_parameters() if param.requires_grad}
    state = raw_model.state_dict()
    return {name: tensor.detach().cpu() for name, tensor in state.items() if name in trainable_names}


def load_trainable_state_dict(model, state):
    raw_model = unwrap_model(model)
    incompatible = raw_model.load_state_dict(state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if key in state]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable keys while loading checkpoint: {unexpected[:10]}")


def count_parameters(model):
    raw_model = unwrap_model(model)
    total = sum(param.numel() for param in raw_model.parameters())
    trainable = sum(param.numel() for param in raw_model.parameters() if param.requires_grad)
    return total, trainable


def build_model(args, classnames, device):
    from transformers import CLIPModel, CLIPTokenizerFast

    tokenizer = CLIPTokenizerFast.from_pretrained(args.model_name)
    clip_model = CLIPModel.from_pretrained(args.model_name)
    input_ids, attention_mask, prompts = build_class_tokens(tokenizer, classnames, args.prompt_template)
    model = DualLoRACLIP(
        clip_model=clip_model,
        class_input_ids=input_ids,
        class_attention_mask=attention_mask,
        vision_layers=args.vision_layers,
        text_layers=args.text_layers,
        vision_rank=args.vision_rank,
        text_rank=args.text_rank,
        vision_alpha=args.vision_alpha,
        text_alpha=args.text_alpha,
        lora_dropout=args.lora_dropout,
        targets=args.lora_targets,
    )
    model.to(device)
    if torch.cuda.device_count() > 1 and args.data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    return model, prompts


def make_optimizer_and_scheduler(args, model, steps_per_epoch):
    from transformers import get_cosine_schedule_with_warmup

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def compute_metrics(y_true, y_pred, losses, num_classes):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels_all = np.arange(num_classes)
    support = np.bincount(y_true, minlength=num_classes)
    present_labels = labels_all[support > 0]
    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels_all, zero_division=0)
    return Metrics(
        loss=float(np.mean(losses)) if losses else 0.0,
        accuracy=100.0 * float(accuracy_score(y_true, y_pred)),
        macro_f1_present=100.0 * float(f1_score(y_true, y_pred, labels=present_labels, average="macro", zero_division=0)),
        macro_f1_all=100.0 * float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)),
        balanced_accuracy_present=100.0 * float(np.mean(recall[present_labels])),
        n=int(len(y_true)),
        n_present_classes=int(len(present_labels)),
    )


@torch.no_grad()
def evaluate(model, loader, device, num_classes, split_name, amp, lora_scale=1.0, ensemble_alphas=None):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    if ensemble_alphas is None:
        ensemble_alphas = [lora_scale]

    progress = tqdm(loader, desc=f"eval/{split_name}", dynamic_ncols=True, leave=False)
    for pixel_values, labels in progress:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits_sum = None
        for alpha in ensemble_alphas:
            set_lora_scale(model, alpha)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(pixel_values).float()
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits = logits_sum / float(len(ensemble_alphas))
        loss = F.cross_entropy(logits, labels)
        pred = logits.argmax(dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        progress.set_postfix(loss=f"{np.mean(losses):.4f}")

    metrics = compute_metrics(y_true, y_pred, losses, num_classes)
    set_lora_scale(model, 1.0)
    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, args, wandb_run):
    model.train()
    set_lora_scale(model, 1.0)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    progress = tqdm(loader, desc=f"train/epoch_{epoch}", dynamic_ncols=True)

    for step, (pixel_values, labels) in enumerate(progress, start=1):
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = int(labels.numel())
        running_loss += float(loss.item()) * batch_size
        running_correct += int(logits.detach().argmax(dim=1).eq(labels).sum().item())
        running_total += batch_size
        lr = scheduler.get_last_lr()[0]
        avg_loss = running_loss / max(1, running_total)
        avg_acc = 100.0 * running_correct / max(1, running_total)
        postfix = {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}", "lr": f"{lr:.2e}"}
        if torch.cuda.is_available():
            postfix["mem_gb"] = f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}"
        progress.set_postfix(postfix)
        if wandb_run is not None and (step == 1 or step % 20 == 0):
            wandb_run.log(
                {
                    "train/loss_step": float(loss.item()),
                    "train/loss_running": avg_loss,
                    "train/acc_running": avg_acc,
                    "train/lr": lr,
                    "epoch": epoch,
                }
            )

    return {"loss": running_loss / max(1, running_total), "accuracy": 100.0 * running_correct / max(1, running_total)}


def save_checkpoint(path, model, args, metrics, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "trainable_state_dict": trainable_state_dict(model),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    load_trainable_state_dict(model, checkpoint["trainable_state_dict"])
    return checkpoint


def print_metrics(prefix, metrics):
    print(
        f"{prefix}: loss={metrics.loss:.4f} acc={metrics.accuracy:.2f} "
        f"macro_f1_present={metrics.macro_f1_present:.2f} "
        f"macro_f1_all={metrics.macro_f1_all:.2f} "
        f"bal_acc_present={metrics.balanced_accuracy_present:.2f} "
        f"n={metrics.n} present_classes={metrics.n_present_classes}",
        flush=True,
    )


def init_wandb(args, extra_config):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    run_name = args.wandb_run_name or f"dual_lora_ce_seed{args.seed}_{int(time.time())}"
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={**vars(args), **extra_config},
        mode=args.wandb_mode,
    )


def log_metrics(wandb_run, prefix, metrics, epoch=None):
    if wandb_run is None:
        return
    payload = {f"{prefix}/{key}": value for key, value in asdict(metrics).items()}
    if epoch is not None:
        payload["epoch"] = epoch
    wandb_run.log(payload)


def write_results(output_dir, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved results: {csv_path}", flush=True)
    print(f"Saved results JSON: {json_path}", flush=True)


def metrics_row(stage, split, metrics, alpha=None, ensemble_alphas=None):
    row = {"stage": stage, "split": split, **asdict(metrics)}
    if alpha is not None:
        row["alpha"] = alpha
    if ensemble_alphas is not None:
        row["ensemble_alphas"] = ",".join(str(alpha) for alpha in ensemble_alphas)
    return row


def main():
    args = parse_args()
    ensure_optional_dependencies(args.auto_install, args.wandb_mode)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(args.data_root)
    metadata = load_metadata(dataset_root)
    filename_col = find_filename_column(metadata)
    image_root = Path(args.image_root) if args.image_root else dataset_root / "train"
    num_classes = int(metadata["y"].max()) + 1
    classnames = load_classnames(dataset_root, args.categories_csv, num_classes, args.empty_class_name)
    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Image root: {image_root}", flush=True)
    print(f"Filename column: {filename_col}", flush=True)
    print(f"Num classes: {num_classes}", flush=True)
    print("Split counts:", flush=True)
    print(metadata["split"].value_counts().to_string(), flush=True)

    train_loader, eval_loaders, split_rows = build_loaders(args, metadata, dataset_root, image_root, filename_col, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; cuda_devices={torch.cuda.device_count()}", flush=True)

    model, prompts = build_model(args, classnames, device)
    total_params, trainable_params = count_parameters(model)
    raw_model = unwrap_model(model)
    print(f"Injected LoRA modules: {raw_model.injected_modules}", flush=True)
    print(f"Params: total={total_params:,} trainable={trainable_params:,} ({100.0 * trainable_params / total_params:.4f}%)", flush=True)
    print("Prompt examples:", flush=True)
    for prompt in prompts[:5]:
        print(f"  {prompt}", flush=True)

    wandb_run = init_wandb(
        args,
        {
            "dataset_root": str(dataset_root),
            "image_root": str(image_root),
            "num_classes": num_classes,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "lora_modules": raw_model.injected_modules,
            "split_sizes": {name: len(rows) for name, rows in split_rows.items()},
            "train_size": len(train_loader.dataset),
        },
    )

    optimizer, scheduler = make_optimizer_and_scheduler(args, model, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_metric = -1.0
    best_ckpt = output_dir / "best_ood_val.pt"
    last_ckpt = output_dir / "last.pt"
    rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args, wandb_run)
        print(f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.2f}", flush=True)
        if wandb_run is not None:
            wandb_run.log({"train/loss_epoch": train_metrics["loss"], "train/accuracy_epoch": train_metrics["accuracy"], "epoch": epoch})

        eval_epoch_metrics = {}
        for split_name in ["id_val", "ood_val"]:
            metrics = evaluate(model, eval_loaders[split_name], device, num_classes, split_name, args.amp, lora_scale=1.0)
            eval_epoch_metrics[split_name] = metrics
            print_metrics(f"epoch {epoch} {split_name}", metrics)
            log_metrics(wandb_run, split_name, metrics, epoch=epoch)
            rows.append(metrics_row("epoch", split_name, metrics, alpha=1.0) | {"epoch": epoch})

        current = eval_epoch_metrics["ood_val"].macro_f1_present
        save_checkpoint(last_ckpt, model, args, {name: asdict(metric) for name, metric in eval_epoch_metrics.items()}, epoch)
        if current > best_metric:
            best_metric = current
            save_checkpoint(best_ckpt, model, args, {name: asdict(metric) for name, metric in eval_epoch_metrics.items()}, epoch)
            print(f"New best ood_val macro_f1_present={best_metric:.2f}; saved {best_ckpt}", flush=True)

    print(f"Loading best checkpoint: {best_ckpt}", flush=True)
    best_payload = load_checkpoint(best_ckpt, model, device)
    print(f"Best checkpoint epoch: {best_payload['epoch']}", flush=True)

    print("Final evaluation at alpha=1.0", flush=True)
    for split_name in ["id_val", "ood_val", "id_test", "ood_test"]:
        metrics = evaluate(model, eval_loaders[split_name], device, num_classes, split_name, args.amp, lora_scale=1.0)
        print_metrics(f"final_alpha1 {split_name}", metrics)
        log_metrics(wandb_run, f"final_alpha1/{split_name}", metrics)
        rows.append(metrics_row("final_alpha1", split_name, metrics, alpha=1.0))

    wise_alphas = parse_float_list(args.wise_alphas)
    wise_val_rows = []
    print("WiSE-FT alpha sweep on id_val and ood_val", flush=True)
    for alpha in wise_alphas:
        id_metrics = evaluate(model, eval_loaders["id_val"], device, num_classes, f"id_val_a{alpha:g}", args.amp, lora_scale=alpha)
        ood_metrics = evaluate(model, eval_loaders["ood_val"], device, num_classes, f"ood_val_a{alpha:g}", args.amp, lora_scale=alpha)
        print_metrics(f"wise alpha={alpha:g} id_val", id_metrics)
        print_metrics(f"wise alpha={alpha:g} ood_val", ood_metrics)
        log_metrics(wandb_run, f"wise/id_val_a{alpha:g}", id_metrics)
        log_metrics(wandb_run, f"wise/ood_val_a{alpha:g}", ood_metrics)
        rows.append(metrics_row("wise_sweep", "id_val", id_metrics, alpha=alpha))
        rows.append(metrics_row("wise_sweep", "ood_val", ood_metrics, alpha=alpha))
        select_metric = ood_metrics.macro_f1_present if args.wise_select == "ood_val" else id_metrics.macro_f1_present
        wise_val_rows.append({"alpha": alpha, "select_metric": select_metric, "id_val": id_metrics, "ood_val": ood_metrics})

    wise_val_rows = sorted(wise_val_rows, key=lambda row: row["select_metric"], reverse=True)
    best_alpha = wise_val_rows[0]["alpha"]
    print(f"Selected WiSE alpha={best_alpha:g} by {args.wise_select} macro_f1_present={wise_val_rows[0]['select_metric']:.2f}", flush=True)
    for split_name in ["id_test", "ood_test"]:
        metrics = evaluate(model, eval_loaders[split_name], device, num_classes, f"{split_name}_wise_best", args.amp, lora_scale=best_alpha)
        print_metrics(f"wise_best alpha={best_alpha:g} {split_name}", metrics)
        log_metrics(wandb_run, f"wise_best/{split_name}", metrics)
        rows.append(metrics_row("wise_best", split_name, metrics, alpha=best_alpha))

    top_k = max(1, min(args.wise_ensemble_top_k, len(wise_val_rows)))
    ensemble_alphas = [row["alpha"] for row in wise_val_rows[:top_k]]
    print(f"WiSE-FT logit ensemble alphas: {ensemble_alphas}", flush=True)
    for split_name in ["id_val", "ood_val", "id_test", "ood_test"]:
        metrics = evaluate(
            model,
            eval_loaders[split_name],
            device,
            num_classes,
            f"{split_name}_wise_ensemble",
            args.amp,
            ensemble_alphas=ensemble_alphas,
        )
        print_metrics(f"wise_ensemble {split_name}", metrics)
        log_metrics(wandb_run, f"wise_ensemble/{split_name}", metrics)
        rows.append(metrics_row("wise_ensemble", split_name, metrics, ensemble_alphas=ensemble_alphas))

    write_results(output_dir, rows)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
