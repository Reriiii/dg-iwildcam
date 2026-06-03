import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


LOG_BATCH_PATTERN = re.compile(r"epoch \[(\d+)/(\d+)\] batch \[(\d+)/(\d+)\]")


def infer_num_batches(log_path):
    if not log_path.exists():
        return None

    num_batches = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            match = LOG_BATCH_PATTERN.search(line)
            if match:
                num_batches = int(match.group(4))
    return num_batches


def load_scalars(event_path):
    accumulator = event_accumulator.EventAccumulator(
        str(event_path), size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    expected = {
        "train/loss": "loss",
        "train/acc": "acc",
        "train/lr": "lr",
    }

    frames = []
    for tag, name in expected.items():
        if tag not in available:
            continue
        rows = [
            {"step": item.step, name: item.value, f"{name}_wall_time": item.wall_time}
            for item in accumulator.Scalars(tag)
        ]
        frames.append(pd.DataFrame(rows))

    if not frames:
        raise RuntimeError(f"No expected train scalars found in {event_path}")

    history = frames[0]
    for frame in frames[1:]:
        history = history.merge(frame, on="step", how="outer")

    history = history.sort_values("step").reset_index(drop=True)
    return history


def summarize_by_epoch(history, num_batches):
    history = history.copy()
    history["epoch"] = history["step"] // num_batches + 1
    history["epoch_progress"] = (history["step"] + 1) / num_batches

    rows = []
    for epoch, group in history.groupby("epoch", sort=True):
        last = group.iloc[-1]
        row = {
            "epoch": int(epoch),
            "last_step": int(last["step"]),
            "loss_end_avg": last.get("loss"),
            "acc_end_avg": last.get("acc"),
            "lr_end": last.get("lr"),
        }
        if "loss" in group:
            row["loss_min_running_avg"] = group["loss"].min()
            row["loss_max_running_avg"] = group["loss"].max()
        if "acc" in group:
            row["acc_min_running_avg"] = group["acc"].min()
            row["acc_max_running_avg"] = group["acc"].max()
        rows.append(row)

    return history, pd.DataFrame(rows)


def plot_history(history, epoch_summary, output_path, title):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    x = history["epoch_progress"]

    if "loss" in history:
        axes[0].plot(x, history["loss"], linewidth=1.0, alpha=0.75, label="running avg")
        axes[0].plot(
            epoch_summary["epoch"],
            epoch_summary["loss_end_avg"],
            marker="o",
            linewidth=1.5,
            label="epoch end",
        )
        axes[0].set_ylabel("Train loss")
        axes[0].legend(loc="upper right")

    if "acc" in history:
        axes[1].plot(x, history["acc"], linewidth=1.0, alpha=0.75, label="running avg")
        axes[1].plot(
            epoch_summary["epoch"],
            epoch_summary["acc_end_avg"],
            marker="o",
            linewidth=1.5,
            label="epoch end",
        )
        axes[1].set_ylabel("Train acc (%)")
        axes[1].legend(loc="lower right")

    if "lr" in history:
        axes[2].plot(x, history["lr"], linewidth=1.0)
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")

    axes[2].set_xlabel("Epoch")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to a CoOp train output directory")
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    event_files = sorted((run_dir / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir / 'tensorboard'}")
    event_path = event_files[-1]

    num_batches = args.num_batches or infer_num_batches(run_dir / "log.txt")
    if not num_batches:
        raise RuntimeError("Could not infer num_batches; pass --num-batches explicitly")

    history = load_scalars(event_path)
    history, epoch_summary = summarize_by_epoch(history, num_batches)

    out_dir = run_dir / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "train_history.csv"
    epoch_path = out_dir / "train_epoch_summary.csv"
    plot_path = out_dir / "train_history.png"

    history.to_csv(history_path, index=False)
    epoch_summary.to_csv(epoch_path, index=False)
    title = args.title or run_dir.name
    plot_history(history, epoch_summary, plot_path, title)

    print(f"Saved history CSV: {history_path}")
    print(f"Saved epoch summary: {epoch_path}")
    print(f"Saved plot: {plot_path}")
    print(epoch_summary.to_string(index=False))


if __name__ == "__main__":
    main()
