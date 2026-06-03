$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\WildCoOpA\vit_b16_iwildcam_adjusted_16shot.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\wildcoop_a_iwildcam_vitb16_adjusted_16shot_seed1"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$maxEpoch = 20
$ShotDir = Join-Path $OutputRoot "shot_$shots"
$TrainDir = Join-Path $ShotDir "train"
$FinalCkpt = Join-Path $TrainDir "wildcoop_a\model.pth.tar-$maxEpoch"
$splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "id_test"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"},
    @{Name = "ood_test"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

if (-not (Test-Path -LiteralPath $FinalCkpt)) {
    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer WildCoOpA `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $TrainDir `
        --resume $TrainDir `
        DATASET.NUM_SHOTS $shots `
        TEST.NO_TEST True `
        OPTIM.MAX_EPOCH $maxEpoch `
        TRAIN.CHECKPOINT_FREQ 5 `
        TRAIN.PRINT_FREQ 20 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TRAIN_X.BATCH_SIZE 64 `
        DATALOADER.TEST.BATCH_SIZE 512
} else {
    "Skipping train because checkpoint exists: $FinalCkpt"
}

foreach ($split in $splits) {
    $EvalDir = Join-Path $ShotDir ("eval_" + $split.Name)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $($split.Name) because result exists: $EvalResult"
        continue
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer WildCoOpA `
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
