param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_text_lora_probe"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_text_lora_probe_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$maxEpoch = 10
$ShotDir = Join-Path $OutputRoot "shot_$shots"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing 16-shot text-LoRA probe output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing 16-shot text-LoRA probe summary: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
}

$Variants = @(
    @{Name = "prompt_only"; Extra = @("TRAINER.COOPLORA.LAYERS", "0", "TRAINER.COOPLORA.TEXT_LAYERS", "0")},
    @{Name = "text_lora_l8"; Extra = @("TRAINER.COOPLORA.LAYERS", "0", "TRAINER.COOPLORA.TEXT_LAYERS", "8")},
    @{Name = "dual_lora_l8"; Extra = @("TRAINER.COOPLORA.LAYERS", "8", "TRAINER.COOPLORA.TEXT_LAYERS", "8")}
)

function Invoke-VariantTrain {
    param([hashtable]$Variant)

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\train")
    $FinalCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"
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
        --config-file $TrainerConfig `
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

function Invoke-VariantEval {
    param(
        [hashtable]$Variant,
        [string]$SplitName,
        [string]$TestSplit,
        [string]$IWildCamTestSplit
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\train")
    $EvalDir = Join-Path $ShotDir ("eval_" + $SplitName + "_" + $Variant.Name)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    $ExtraArgs = @($Variant.Extra)
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $SplitName $($Variant.Name) because result exists: $EvalResult"
        return
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRA `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $EvalDir `
        --eval-only `
        --model-dir $TrainDir `
        --load-epoch $maxEpoch `
        DATASET.NUM_SHOTS $shots `
        TEST.SPLIT $TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $IWildCamTestSplit `
        TRAINER.COOPLORA.KL_WEIGHT 0.0 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512 `
        @ExtraArgs
}

foreach ($variant in $Variants) {
    Invoke-VariantTrain -Variant $variant
    Invoke-VariantEval -Variant $variant -SplitName "id_val" -TestSplit "val" -IWildCamTestSplit "val"
    Invoke-VariantEval -Variant $variant -SplitName "ood_val" -TestSplit "test" -IWildCamTestSplit "val"
    Invoke-VariantEval -Variant $variant -SplitName "id_test" -TestSplit "test" -IWildCamTestSplit "id_test"
    Invoke-VariantEval -Variant $variant -SplitName "ood_test" -TestSplit "test" -IWildCamTestSplit "test"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
