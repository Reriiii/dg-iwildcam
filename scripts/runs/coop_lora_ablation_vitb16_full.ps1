$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_ablation"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$maxEpoch = 10
$experiments = @(
    @{Name = "r8_l4"; Config = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r8_l4.yaml"},
    @{Name = "r4_l8"; Config = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"},
    @{Name = "r8_l8"; Config = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r8_l8.yaml"}
)

foreach ($experiment in $experiments) {
    $ShotDir = Join-Path $OutputRoot $experiment.Name
    $TrainDir = Join-Path $ShotDir "train"
    $FinalCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"

    if (-not (Test-Path -LiteralPath $FinalCkpt)) {
        python -u "$CoOpRoot\train.py" `
            --root "$RepoRoot\data" `
            --seed $seed `
            --trainer CoOpLoRA `
            --dataset-config-file $DatasetConfig `
            --config-file $experiment.Config `
            --output-dir $TrainDir `
            --resume $TrainDir `
            DATASET.NUM_SHOTS -1 `
            TEST.NO_TEST True `
            OPTIM.MAX_EPOCH $maxEpoch `
            TRAIN.CHECKPOINT_FREQ 5 `
            TRAIN.PRINT_FREQ 100 `
            DATALOADER.NUM_WORKERS 4 `
            DATALOADER.TRAIN_X.BATCH_SIZE 64 `
            DATALOADER.TEST.BATCH_SIZE 512
    } else {
        "Skipping train $($experiment.Name) because checkpoint exists: $FinalCkpt"
    }

    $EvalDir = Join-Path $ShotDir "eval_ood_val"
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping ood_val eval $($experiment.Name) because result exists: $EvalResult"
        continue
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRA `
        --dataset-config-file $DatasetConfig `
        --config-file $experiment.Config `
        --output-dir $EvalDir `
        --eval-only `
        --model-dir $TrainDir `
        --load-epoch $maxEpoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT test `
        DATASET.IWILDCAM_TEST_SPLIT val `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}
