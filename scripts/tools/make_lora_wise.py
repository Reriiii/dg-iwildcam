import argparse
import sys
from pathlib import Path

import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_clip_to_cpu(backbone_name, coop_root):
    sys.path.insert(0, str(coop_root))
    from clip import clip

    model_path = clip._download(clip._MODELS[backbone_name])
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return clip.build_model(state_dict or model.state_dict())


def get_initial_ctx(backbone_name, ctx_init, dtype, coop_root):
    sys.path.insert(0, str(coop_root))
    from clip import clip

    clip_model = load_clip_to_cpu(backbone_name, coop_root)
    ctx_init = ctx_init.replace("_", " ")
    n_ctx = len(ctx_init.split(" "))
    prompt = clip.tokenize(ctx_init)
    with torch.no_grad():
        embedding = clip_model.token_embedding(prompt).type(dtype)
    return embedding[0, 1 : 1 + n_ctx, :].cpu()


def scale_state_dict(state_dict, init_ctx, prompt_alpha, lora_alpha):
    scaled = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            scaled[key] = value
            continue

        new_value = value.detach().clone()
        if key == "prompt_learner.ctx":
            if init_ctx.shape != value.shape:
                raise ValueError(f"Initial ctx shape {tuple(init_ctx.shape)} does not match checkpoint {tuple(value.shape)}")
            if prompt_alpha == 1.0:
                new_value = value.cpu()
            elif prompt_alpha == 0.0:
                new_value = init_ctx.to(dtype=value.dtype)
            else:
                init = init_ctx.float()
                new_value = init + prompt_alpha * (value.cpu().float() - init)
        elif key.endswith(".in_B") or key.endswith(".out_B"):
            new_value = value.cpu() if lora_alpha == 1.0 else value.cpu() * lora_alpha

        scaled[key] = new_value.to(dtype=value.dtype) if torch.is_floating_point(value) else new_value
    return scaled


def format_alpha(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser(description="Create a CoOpLoRA WiSE-style checkpoint by scaling prompt and LoRA deltas.")
    parser.add_argument("--source", required=True, help="Fine-tuned CoOpLoRA checkpoint")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--alpha", type=float, required=True, help="Default alpha for prompt and LoRA deltas")
    parser.add_argument("--prompt-alpha", type=float, default=None, help="Prompt delta alpha; defaults to --alpha")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA delta alpha; defaults to --alpha")
    parser.add_argument("--ctx-init", default="a camera trap photo of")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--coop-root", default="external/CoOp")
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()

    prompt_alpha = args.alpha if args.prompt_alpha is None else args.prompt_alpha
    lora_alpha = args.alpha if args.lora_alpha is None else args.lora_alpha
    for name, value in [("alpha", args.alpha), ("prompt_alpha", prompt_alpha), ("lora_alpha", lora_alpha)]:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    source = Path(args.source)
    output = Path(args.output)
    coop_root = Path(args.coop_root)

    checkpoint = load_checkpoint(source)
    state_dict = checkpoint["state_dict"]
    prompt_ctx = state_dict["prompt_learner.ctx"]
    init_ctx = get_initial_ctx(args.backbone, args.ctx_init, prompt_ctx.dtype, coop_root)
    scaled = scale_state_dict(state_dict, init_ctx, prompt_alpha, lora_alpha)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": scaled,
            "epoch": args.epoch,
            "optimizer": None,
            "scheduler": None,
            "val_result": -1.0,
            "wise": {
                "source": str(source),
                "alpha": format_alpha(args.alpha),
                "prompt_alpha": format_alpha(prompt_alpha),
                "lora_alpha": format_alpha(lora_alpha),
                "ctx_init": args.ctx_init,
                "backbone": args.backbone,
            },
        },
        output,
    )
    checkpoint_file = output.parent / "checkpoint"
    checkpoint_file.write_text(output.name + "\n", encoding="utf-8")
    print(f"Saved CoOpLoRA-WiSE checkpoint: {output}")
    print(f"alpha={format_alpha(args.alpha)} prompt_alpha={format_alpha(prompt_alpha)} lora_alpha={format_alpha(lora_alpha)}")


if __name__ == "__main__":
    main()
