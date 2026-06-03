param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\_smoke_coop_lora_sequence_cap"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing SequenceCap smoke output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
}

python -u "$CoOpRoot\train.py" `
    --root "$RepoRoot\data" `
    --seed 1 `
    --trainer CoOpLoRA `
    --dataset-config-file $DatasetConfig `
    --config-file $TrainerConfig `
    --output-dir $OutputRoot `
    DATASET.NUM_SHOTS 16 `
    TEST.NO_TEST True `
    OPTIM.MAX_EPOCH 1 `
    TRAIN.CHECKPOINT_FREQ 1 `
    TRAIN.PRINT_FREQ 20 `
    DATALOADER.NUM_WORKERS 4 `
    DATALOADER.TRAIN_X.BATCH_SIZE 64 `
    DATALOADER.TRAIN_X.SAMPLER SequenceCapSampler `
    DATALOADER.TRAIN_X.SEQ_MAX_PER_SEQUENCE 8 `
    DATALOADER.TEST.BATCH_SIZE 512

$FinalCkpt = Join-Path $OutputRoot "coop_lora\model.pth.tar-1"
if (-not (Test-Path -LiteralPath $FinalCkpt)) {
    throw "Missing smoke checkpoint: $FinalCkpt"
}

"SequenceCap smoke passed: $FinalCkpt"
