$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\WildCoOpA\vit_b16_iwildcam_adjusted_16shot.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\wildcoop_a_iwildcam_vitb16_16shot_quick_ablation_seed1"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$maxEpoch = 20

$cases = @(
    @{
        Name = "freq_only"
        Extra = @(
            "TRAINER.WILDCOOPA.USE_FREQ_PROMPT", "True"
        )
    },
    @{
        Name = "manual_only"
        Extra = @(
            "TRAINER.WILDCOOPA.USE_MANUAL_PROTO", "True",
            "TRAINER.WILDCOOPA.LAMBDA_MIX", "0.7",
            "TRAINER.WILDCOOPA.LAMBDA_PROTO", "0.05"
        )
    },
    @{
        Name = "adapter_only"
        Extra = @(
            "TRAINER.WILDCOOPA.USE_VISUAL_ADAPTER", "True",
            "TRAINER.WILDCOOPA.LAMBDA_ADAPTER", "0.001"
        )
    },
    @{
        Name = "weighted_ce"
        Extra = @(
            "TRAINER.WILDCOOPA.USE_CLASS_BALANCED_FOCAL", "True",
            "TRAINER.WILDCOOPA.FOCAL_GAMMA", "0.0"
        )
    }
)

foreach ($case in $cases) {
    $CaseRoot = Join-Path $OutputRoot $case.Name
    $ShotDir = Join-Path $CaseRoot "shot_$shots"
    $TrainDir = Join-Path $ShotDir "train"
    $EvalDir = Join-Path $ShotDir "eval_ood_val"
    $FinalCkpt = Join-Path $TrainDir "wildcoop_a\model.pth.tar-$maxEpoch"
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"

    "=== WildCoOp-A quick ablation: $($case.Name) ==="

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
            DATALOADER.TEST.BATCH_SIZE 512 `
            @($case.Extra)
    } else {
        "Skipping train because checkpoint exists: $FinalCkpt"
    }

    if (-not (Test-Path -LiteralPath $EvalResult)) {
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
            TEST.SPLIT test `
            DATASET.IWILDCAM_TEST_SPLIT val `
            DATALOADER.NUM_WORKERS 4 `
            DATALOADER.TEST.BATCH_SIZE 512 `
            @($case.Extra)
    } else {
        "Skipping ood_val eval because result exists: $EvalResult"
    }
}
