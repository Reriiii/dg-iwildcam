import argparse
from pathlib import Path

import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_weights(weights):
    total = sum(weights)
    if total == 0:
        raise ValueError("Weights must not sum to zero")
    return [weight / total for weight in weights]


def average_state_dicts(state_dicts, weights):
    keys = set(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        if set(state_dict.keys()) != keys:
            missing = sorted(keys - set(state_dict.keys()))[:5]
            extra = sorted(set(state_dict.keys()) - keys)[:5]
            raise ValueError(f"State dict keys differ, missing={missing}, extra={extra}")

    averaged = {}
    for key in state_dicts[0].keys():
        value = state_dicts[0][key]
        if torch.is_tensor(value) and torch.is_floating_point(value):
            acc = torch.zeros_like(value, dtype=torch.float32)
            for state_dict, weight in zip(state_dicts, weights):
                acc += state_dict[key].float() * weight
            averaged[key] = acc.to(dtype=value.dtype)
        else:
            averaged[key] = value
    return averaged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()

    if len(args.sources) != len(args.weights):
        raise ValueError("--sources and --weights must have the same length")

    weights = normalize_weights(args.weights)
    checkpoints = [load_checkpoint(Path(source)) for source in args.sources]
    state_dicts = [checkpoint["state_dict"] for checkpoint in checkpoints]
    averaged = average_state_dicts(state_dicts, weights)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": averaged,
            "epoch": args.epoch,
            "optimizer": None,
            "scheduler": None,
            "val_result": -1.0,
        },
        output,
    )
    checkpoint_file = output.parent / "checkpoint"
    checkpoint_file.write_text(output.name + "\n", encoding="utf-8")
    print(f"Saved soup checkpoint: {output}")
    print("weights=" + ",".join(f"{weight:.4f}" for weight in weights))


if __name__ == "__main__":
    main()
