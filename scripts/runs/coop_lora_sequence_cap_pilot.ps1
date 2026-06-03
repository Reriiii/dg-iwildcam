param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_sequence_cap_pilot"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_sequence_cap_pilot_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$variants = @(
    @{Name = "ce_3ep_random"; Epochs = 3; Sampler = "RandomSampler"; SeqMax = 0},
    @{Name = "seqcap8_3ep"; Epochs = 3; Sampler = "SequenceCapSampler"; SeqMax = 8},
    @{Name = "seqcap5_4ep"; Epochs = 4; Sampler = "SequenceCapSampler"; SeqMax = 5}
)
$splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"}
)

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing SequenceCap pilot output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing SequenceCap pilot summary: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
}

function Invoke-TrainVariant {
    param([hashtable]$Variant)

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$seed\train")
    $FinalCkpt = Join-Path $TrainDir ("coop_lora\model.pth.tar-" + $Variant.Epochs)
    if (Test-Path -LiteralPath $FinalCkpt) {
        "Skipping train $($Variant.Name) because checkpoint exists: $FinalCkpt"
        return
    }

    if ($Variant.Sampler -eq "SequenceCapSampler") {
        python -u "$CoOpRoot\train.py" `
            --root "$RepoRoot\data" `
            --seed $seed `
            --trainer CoOpLoRA `
            --dataset-config-file $DatasetConfig `
            --config-file $TrainerConfig `
            --output-dir $TrainDir `
            DATASET.NUM_SHOTS -1 `
            TEST.NO_TEST True `
            OPTIM.MAX_EPOCH $Variant.Epochs `
            TRAIN.CHECKPOINT_FREQ $Variant.Epochs `
            TRAIN.PRINT_FREQ 100 `
            DATALOADER.NUM_WORKERS 4 `
            DATALOADER.TRAIN_X.BATCH_SIZE 64 `
            DATALOADER.TRAIN_X.SAMPLER SequenceCapSampler `
            DATALOADER.TRAIN_X.SEQ_MAX_PER_SEQUENCE $Variant.SeqMax `
            DATALOADER.TEST.BATCH_SIZE 512
    } else {
        python -u "$CoOpRoot\train.py" `
            --root "$RepoRoot\data" `
            --seed $seed `
            --trainer CoOpLoRA `
            --dataset-config-file $DatasetConfig `
            --config-file $TrainerConfig `
            --output-dir $TrainDir `
            DATASET.NUM_SHOTS -1 `
            TEST.NO_TEST True `
            OPTIM.MAX_EPOCH $Variant.Epochs `
            TRAIN.CHECKPOINT_FREQ $Variant.Epochs `
            TRAIN.PRINT_FREQ 100 `
            DATALOADER.NUM_WORKERS 4 `
            DATALOADER.TRAIN_X.BATCH_SIZE 64 `
            DATALOADER.TRAIN_X.SAMPLER RandomSampler `
            DATALOADER.TEST.BATCH_SIZE 512
    }

    if (-not (Test-Path -LiteralPath $FinalCkpt)) {
        throw "Missing trained checkpoint for $($Variant.Name): $FinalCkpt"
    }
}

function Invoke-EvalVariant {
    param(
        [hashtable]$Variant,
        [hashtable]$Split
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$seed\train")
    $EvalDir = Join-Path $OutputRoot ("shot_full\eval_" + $Split.Name + "_" + $Variant.Name)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $($Split.Name) $($Variant.Name) because result exists: $EvalResult"
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
        --load-epoch $Variant.Epochs `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT $Split.TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $Split.IWildCamTestSplit `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

foreach ($variant in $variants) {
    Invoke-TrainVariant -Variant $variant
    foreach ($split in $splits) {
        Invoke-EvalVariant -Variant $variant -Split $split
    }
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
