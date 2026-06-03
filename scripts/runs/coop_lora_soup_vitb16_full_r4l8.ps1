$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_soup"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_soup_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$soupEpoch = 1
$ShotDir = Join-Path $OutputRoot "shot_full"

$A10 = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_ablation\r4_l8\train\coop_lora\model.pth.tar-10"
$A5 = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_ablation\r4_l8\train\coop_lora\model.pth.tar-5"
$B10 = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_20ep_r4_l8_clean\shot_full\train\coop_lora\model.pth.tar-10"
$B15 = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_20ep_r4_l8_clean\shot_full\train\coop_lora\model.pth.tar-15"
$B20 = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_20ep_r4_l8_clean\shot_full\train\coop_lora\model.pth.tar-20"

$cases = @(
    @{Name = "a10_a5_a0p25"; Sources = @($A10, $A5); Weights = @(0.25, 0.75)},
    @{Name = "a10_a5_a0p50"; Sources = @($A10, $A5); Weights = @(0.50, 0.50)},
    @{Name = "a10_a5_a0p75"; Sources = @($A10, $A5); Weights = @(0.75, 0.25)},
    @{Name = "a10_a5_a0p80"; Sources = @($A10, $A5); Weights = @(0.80, 0.20)},
    @{Name = "a10_a5_a0p85"; Sources = @($A10, $A5); Weights = @(0.85, 0.15)},
    @{Name = "a10_a5_a0p90"; Sources = @($A10, $A5); Weights = @(0.90, 0.10)},
    @{Name = "a10_a5_a0p95"; Sources = @($A10, $A5); Weights = @(0.95, 0.05)},
    @{Name = "a10_b10_a0p25"; Sources = @($A10, $B10); Weights = @(0.25, 0.75)},
    @{Name = "a10_b10_a0p50"; Sources = @($A10, $B10); Weights = @(0.50, 0.50)},
    @{Name = "a10_b10_a0p75"; Sources = @($A10, $B10); Weights = @(0.75, 0.25)},
    @{Name = "a10_b20_a0p25"; Sources = @($A10, $B20); Weights = @(0.25, 0.75)},
    @{Name = "a10_b20_a0p50"; Sources = @($A10, $B20); Weights = @(0.50, 0.50)},
    @{Name = "a10_b20_a0p75"; Sources = @($A10, $B20); Weights = @(0.75, 0.25)},
    @{Name = "a10_b10_b15_b20_eq"; Sources = @($A10, $B10, $B15, $B20); Weights = @(1.0, 1.0, 1.0, 1.0)}
)

foreach ($case in $cases) {
    foreach ($source in $case.Sources) {
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Missing source checkpoint for $($case.Name): $source"
        }
    }

    $ModelDir = Join-Path $OutputRoot ("models\" + $case.Name + "\train")
    $SoupCkpt = Join-Path $ModelDir "coop_lora\model.pth.tar-$soupEpoch"
    if (-not (Test-Path -LiteralPath $SoupCkpt)) {
        python "$RepoRoot\scripts\tools\make_lora_soup.py" `
            --sources @($case.Sources) `
            --weights @($case.Weights) `
            --output $SoupCkpt `
            --epoch $soupEpoch
    } else {
        "Skipping soup creation for $($case.Name) because checkpoint exists: $SoupCkpt"
    }

    $EvalDir = Join-Path $ShotDir ("eval_ood_val_" + $case.Name)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping ood_val for $($case.Name) because result exists: $EvalResult"
        continue
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRA `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $EvalDir `
        --eval-only `
        --model-dir $ModelDir `
        --load-epoch $soupEpoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT test `
        DATASET.IWILDCAM_TEST_SPLIT val `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed

$Best = Import-Csv -LiteralPath $SummaryCsv |
    Where-Object { $_.split -like "ood_val_*" } |
    Sort-Object {[double]$_.macro_f1_present} -Descending |
    Select-Object -First 1

if (-not $Best) {
    throw "No soup ood_val rows found in $SummaryCsv"
}

$BestCase = $Best.split.Replace("ood_val_", "")
"Selected soup=$BestCase by ood_val macro_f1_present=$($Best.macro_f1_present)"

$BestModelDir = Join-Path $OutputRoot ("models\" + $BestCase + "\train")
$EvalDir = Join-Path $ShotDir ("eval_ood_test_" + $BestCase)
$EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
if (-not (Test-Path -LiteralPath $EvalResult)) {
    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRA `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $EvalDir `
        --eval-only `
        --model-dir $BestModelDir `
        --load-epoch $soupEpoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT test `
        DATASET.IWILDCAM_TEST_SPLIT test `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
} else {
    "Skipping ood_test for $BestCase because result exists: $EvalResult"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
