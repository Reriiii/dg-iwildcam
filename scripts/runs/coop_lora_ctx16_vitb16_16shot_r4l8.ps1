param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_16shot_ctx16_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_ctx16_r4_l8_seed1"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_ctx16_r4_l8_seed1_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$maxEpoch = 10
$ShotDir = Join-Path $OutputRoot "shot_$shots"
$TrainDir = Join-Path $ShotDir "train"
$FinalCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing output for clean 16-shot run: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing summary for clean 16-shot run: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
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
    DATALOADER.TEST.BATCH_SIZE 512

if (-not (Test-Path -LiteralPath $FinalCkpt)) {
    throw "Missing trained checkpoint: $FinalCkpt"
}

$splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"},
    @{Name = "id_test"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_test"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

foreach ($split in $splits) {
    $EvalDir = Join-Path $ShotDir ("eval_" + $split.Name)
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
        TEST.SPLIT $split.TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $split.IWildCamTestSplit `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
