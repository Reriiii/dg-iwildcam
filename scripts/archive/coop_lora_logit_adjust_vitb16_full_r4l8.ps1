$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$ModelDir = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_ablation\r4_l8\train"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_logit_adjust"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_logit_adjust_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$epoch = 10
$culture = [Globalization.CultureInfo]::InvariantCulture
$tauValues = @(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25)
$ShotDir = Join-Path $OutputRoot "shot_full"
$Ckpt = Join-Path $ModelDir "coop_lora\model.pth.tar-$epoch"

if (-not (Test-Path -LiteralPath $Ckpt)) {
    throw "Missing checkpoint: $Ckpt"
}

foreach ($tau in $tauValues) {
    $tauText = $tau.ToString("0.##", $culture)
    if (-not $tauText.Contains(".")) {
        $tauText = "$tauText.0"
    }
    $tauToken = $tauText.Replace(".", "p")
    $EvalDir = Join-Path $ShotDir ("eval_ood_val_tau_" + $tauToken)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping ood_val tau=$tauText because result exists: $EvalResult"
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
        --load-epoch $epoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT test `
        DATASET.IWILDCAM_TEST_SPLIT val `
        TEST.IWILDCAM_LOGIT_ADJUST_TAU $tauText `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed

$Best = Import-Csv -LiteralPath $SummaryCsv |
    Where-Object { $_.split -like "ood_val_tau_*" } |
    Sort-Object {[double]$_.macro_f1_present} -Descending |
    Select-Object -First 1

if (-not $Best) {
    throw "No ood_val tau rows found in $SummaryCsv"
}

$BestTauToken = $Best.split.Replace("ood_val_tau_", "")
$BestTauText = $BestTauToken.Replace("p", ".")
if (-not $BestTauText.Contains(".")) {
    $BestTauText = "$BestTauText.0"
}
"Selected tau=$BestTauText by ood_val macro_f1_present=$($Best.macro_f1_present)"

$EvalDir = Join-Path $ShotDir ("eval_ood_test_tau_" + $BestTauToken)
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
        --model-dir $ModelDir `
        --load-epoch $epoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT test `
        DATASET.IWILDCAM_TEST_SPLIT test `
        TEST.IWILDCAM_LOGIT_ADJUST_TAU $BestTauText `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
} else {
    "Skipping ood_test tau=$BestTauText because result exists: $EvalResult"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
