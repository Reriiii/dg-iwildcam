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
