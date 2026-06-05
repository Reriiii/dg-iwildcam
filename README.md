# DG-iWildCam CoOpLoRA Experiments

This workspace keeps external code and data separate from local experiment tooling.

## Layout

| path | purpose |
|---|---|
| `data/` | Local iWildCam dataset files |
| `external/` | Third-party repos and patched CoOp/Dassl code |
| `notebooks/` | Exploratory notebooks |
| `paper/` | Reference papers |
| `scripts/runs/` | Active reproducible experiment runners |
| `scripts/tools/` | Utility scripts for summarization, soups, WiSE checkpoints, diagnostics |
| `scripts/archive/` | Old/deprecated runners kept only for provenance |
| `outputs/` | Retained best/source experiment outputs and reports |

## Current Active Runners

| command | purpose |
|---|---|
| `powershell -File scripts/runs/coop_lora_ablation_vitb16_full.ps1` | Reproduce retained CoOpLoRA r4_l8 source checkpoint/evals |
| `powershell -File scripts/runs/coop_lora_wise_vitb16_full_r4l8.ps1` | Run CoOpLoRA-WiSE alpha sweep |
| `powershell -File scripts/runs/coop_lora_soup_vitb16_full_r4l8.ps1` | Run checkpoint soup sweep |
| `powershell -File scripts/runs/coop_lora_cb_prompt_fixed_wise_vitb16_full_r4l8.ps1` | Train class-balanced CoOpLoRA and run prompt-fixed LoRA-WiSE |

## Key Reports

| report | purpose |
|---|---|
| `outputs/iwildcam_paper_style_comparison.md` | Paper-style iWildCam comparison table |
| `outputs/experiment_results_summary.md` | Current concise result summary |
| `outputs/coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_report.md` | WiSE-style interpolation report |
| `outputs/coop_lora_iwildcam_vitb16_full_10ep_r4_l8_soup_report.md` | Checkpoint soup report |

## Current Best

`CoOpLoRA-WiSE r4_l8 alpha=0.80` reaches `37.03` OOD macro-F1 on `ood_test`, compared with FLYP with WiSE at `37.10`.

## Kaggle Dual-LoRA Baseline

Standalone Kaggle runner using HuggingFace CLIP ViT-B/16, visual+text LoRA, CE loss, ID/OOD macro-F1 evaluation, WiSE-FT alpha sweep/logit ensemble, tqdm terminal logging, and wandb logging:

```bash
python scripts/kaggle_dual_lora_iwildcam.py --data-root /kaggle/input/iwildcam-v2/iwildcam_v2.0 --output-dir /kaggle/working/dual_lora_iwildcam --epochs 10 --batch-size 128 --eval-batch-size 256 --wandb-project dg-iwildcam-dual-lora --wandb-mode online
```

Use `--wandb-mode offline` if Kaggle is not logged in to W&B. The script defaults to `DataParallel` when two T4 GPUs are available.
