# Scripts

| folder | purpose |
|---|---|
| `runs/` | Active scripts worth rerunning |
| `tools/` | Reusable Python utilities |
| `archive/` | Deprecated or historical runners |

Run scripts from the repository root so relative output paths resolve consistently.

Example:

```powershell
powershell -File scripts/runs/coop_lora_wise_vitb16_full_r4l8.ps1
```

Kaggle standalone baseline:

```bash
python scripts/kaggle_dual_lora_iwildcam.py --data-root /kaggle/input/iwildcam-v2/iwildcam_v2.0 --output-dir /kaggle/working/dual_lora_iwildcam --epochs 10 --batch-size 128 --eval-batch-size 256 --wandb-mode online
```
