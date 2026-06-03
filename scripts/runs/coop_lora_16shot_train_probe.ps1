param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$BaseTrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$LowCapacityConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l4.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_train_probe"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_train_probe_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$maxEpoch = 10
$ShotDir = Join-Path $OutputRoot "shot_$shots"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing 16-shot train probe output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing 16-shot train probe summary: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
}

$Variants = @(
    @{Name = "kl0p05"; Config = $BaseTrainerConfig; Extra = @("TRAINER.COOPLORA.KL_WEIGHT", "0.05", "TRAINER.COOPLORA.KL_TEMPERATURE", "2.0")},
    @{Name = "kl0p1"; Config = $BaseTrainerConfig; Extra = @("TRAINER.COOPLORA.KL_WEIGHT", "0.1", "TRAINER.COOPLORA.KL_TEMPERATURE", "2.0")},
    @{Name = "kl0p2"; Config = $BaseTrainerConfig; Extra = @("TRAINER.COOPLORA.KL_WEIGHT", "0.2", "TRAINER.COOPLORA.KL_TEMPERATURE", "2.0")},
    @{Name = "r4_l4"; Config = $LowCapacityConfig; Extra = @("TRAINER.COOPLORA.KL_WEIGHT", "0.0")}
)

function Invoke-CoOpLoRATrain {
    param(
        [hashtable]$Variant
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\train")
    $FinalCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"
    $Config = $Variant.Config
    $ExtraArgs = @($Variant.Extra)
    if (Test-Path -LiteralPath $FinalCkpt) {
        "Skipping train $($Variant.Name) because checkpoint exists: $FinalCkpt"
        return
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRA `
        --dataset-config-file $DatasetConfig `
        --config-file $Config `
        --output-dir $TrainDir `
        DATASET.NUM_SHOTS $shots `
        TEST.NO_TEST True `
        OPTIM.MAX_EPOCH $maxEpoch `
        TRAIN.CHECKPOINT_FREQ 5 `
        TRAIN.PRINT_FREQ 20 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TRAIN_X.BATCH_SIZE 32 `
        DATALOADER.TEST.BATCH_SIZE 512 `
        @ExtraArgs

    if (-not (Test-Path -LiteralPath $FinalCkpt)) {
        throw "Missing trained checkpoint for $($Variant.Name): $FinalCkpt"
    }
}

function Invoke-CoOpLoRAEval {
    param(
        [string]$VariantName,
        [string]$Config,
        [string]$EvalName,
        [string]$TestSplit,
        [string]$IWildCamTestSplit
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $VariantName + "\train")
    $EvalDir = Join-Path $ShotDir ("eval_" + $EvalName)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $EvalName because result exists: $EvalResult"
        return
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRA `
        --dataset-config-file $DatasetConfig `
        --config-file $Config `
        --output-dir $EvalDir `
        --eval-only `
        --model-dir $TrainDir `
        --load-epoch $maxEpoch `
        DATASET.NUM_SHOTS $shots `
        TEST.SPLIT $TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $IWildCamTestSplit `
        TRAINER.COOPLORA.KL_WEIGHT 0.0 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

foreach ($variant in $Variants) {
    Invoke-CoOpLoRATrain -Variant $variant
    Invoke-CoOpLoRAEval -VariantName $variant.Name -Config $variant.Config -EvalName ("id_val_" + $variant.Name) -TestSplit "val" -IWildCamTestSplit "val"
    Invoke-CoOpLoRAEval -VariantName $variant.Name -Config $variant.Config -EvalName ("ood_val_" + $variant.Name) -TestSplit "test" -IWildCamTestSplit "val"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed

$BestOodCases = Import-Csv -LiteralPath $SummaryCsv |
    Where-Object { $_.split -like "ood_val_*" } |
    Sort-Object {[double]$_.macro_f1_present} -Descending |
    Select-Object -First 3

foreach ($row in $BestOodCases) {
    $case = $row.split.Replace("ood_val_", "")
    $variant = $Variants | Where-Object { $_.Name -eq $case } | Select-Object -First 1
    if (-not $variant) {
        throw "Could not find variant for selected case: $case"
    }
    Invoke-CoOpLoRAEval -VariantName $variant.Name -Config $variant.Config -EvalName ("id_test_" + $variant.Name) -TestSplit "test" -IWildCamTestSplit "id_test"
    Invoke-CoOpLoRAEval -VariantName $variant.Name -Config $variant.Config -EvalName ("ood_test_" + $variant.Name) -TestSplit "test" -IWildCamTestSplit "test"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
