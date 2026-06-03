$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOp\vit_b16_iwildcam_ctx.yaml"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$shots = 16
$maxEpoch = 20
$seeds = @(1, 2, 3)

foreach ($seed in $seeds) {
    $OutputRoot = Join-Path $RepoRoot "outputs\coop_iwildcam_vitb16_16shot_epoch_sweep_workers0_seed$seed"
    $ShotDir = Join-Path $OutputRoot "shot_$shots"
    $TrainDir = Join-Path $ShotDir "train"
    $FinalCkpt = Join-Path $TrainDir "prompt_learner\model.pth.tar-$maxEpoch"

    "=== CoOp ViT-B/16 16-shot epoch sweep seed $seed ==="

    if (-not (Test-Path -LiteralPath $FinalCkpt)) {
        python -u "$CoOpRoot\train.py" `
            --root "$RepoRoot\data" `
            --seed $seed `
            --trainer CoOp `
            --dataset-config-file $DatasetConfig `
            --config-file $TrainerConfig `
            --output-dir $TrainDir `
            DATASET.NUM_SHOTS $shots `
            TEST.NO_TEST True `
            OPTIM.MAX_EPOCH $maxEpoch `
            TRAIN.CHECKPOINT_FREQ 1 `
            TRAIN.PRINT_FREQ 20 `
            DATALOADER.NUM_WORKERS 0 `
            DATALOADER.TRAIN_X.BATCH_SIZE 32 `
            DATALOADER.TEST.BATCH_SIZE 512
    } else {
        "Skipping train because checkpoint exists: $FinalCkpt"
    }

    foreach ($epoch in 1..$maxEpoch) {
        $EvalDir = Join-Path $ShotDir ("eval_ood_val_epoch_" + $epoch)
        $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
        if (Test-Path -LiteralPath $EvalResult) {
            "Skipping eval ood_val epoch $epoch because result exists: $EvalResult"
            continue
        }

        python -u "$CoOpRoot\train.py" `
            --root "$RepoRoot\data" `
            --seed $seed `
            --trainer CoOp `
            --dataset-config-file $DatasetConfig `
            --config-file $TrainerConfig `
            --output-dir $EvalDir `
            --eval-only `
            --model-dir $TrainDir `
            --load-epoch $epoch `
            DATASET.NUM_SHOTS $shots `
            TEST.SPLIT test `
            DATASET.IWILDCAM_TEST_SPLIT val `
            DATALOADER.NUM_WORKERS 0 `
            DATALOADER.TEST.BATCH_SIZE 512
    }
}
