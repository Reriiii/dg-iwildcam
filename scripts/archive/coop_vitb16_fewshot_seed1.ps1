$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_iwildcam_seed1"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$shotsList = @(1, 4, 16)
$splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "id_test"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"},
    @{Name = "ood_test"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

foreach ($shots in $shotsList) {
    $ShotDir = Join-Path $OutputRoot "shot_$shots"
    $TrainDir = Join-Path $ShotDir "train"
    $FinalCkpt = Join-Path $TrainDir "prompt_learner\model.pth.tar-20"

    if (-not (Test-Path -LiteralPath $FinalCkpt)) {
        python train.py `
            --root "$RepoRoot\data" `
            --seed 1 `
            --trainer CoOp `
            --dataset-config-file configs/datasets/iwildcam.yaml `
            --config-file configs/trainers/CoOp/vit_b16_iwildcam_ctx.yaml `
            --output-dir $TrainDir `
            DATASET.NUM_SHOTS $shots `
            TEST.NO_TEST True `
            DATALOADER.NUM_WORKERS 0 `
            DATALOADER.TRAIN_X.BATCH_SIZE 32 `
            DATALOADER.TEST.BATCH_SIZE 512
    } else {
        "Skipping train shot=$shots because checkpoint exists: $FinalCkpt"
    }

    foreach ($split in $splits) {
        $EvalDir = Join-Path $ShotDir ("eval_" + $split.Name)
        python train.py `
            --root "$RepoRoot\data" `
            --seed 1 `
            --trainer CoOp `
            --dataset-config-file configs/datasets/iwildcam.yaml `
            --config-file configs/trainers/CoOp/vit_b16_iwildcam_ctx.yaml `
            --output-dir $EvalDir `
            --eval-only `
            --model-dir $TrainDir `
            --load-epoch 20 `
            DATASET.NUM_SHOTS $shots `
            TEST.SPLIT $split.TestSplit `
            DATASET.IWILDCAM_TEST_SPLIT $split.IWildCamTestSplit `
            DATALOADER.NUM_WORKERS 0 `
            DATALOADER.TEST.BATCH_SIZE 512
    }
}
