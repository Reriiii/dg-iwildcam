$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRASigLIP\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_siglip_iwildcam_vitb16_full_10ep_r4_l8_clean2"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_siglip_iwildcam_vitb16_full_10ep_r4_l8_clean2_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$maxEpoch = 10
$ShotDir = Join-Path $OutputRoot "shot_full"
$TrainDir = Join-Path $ShotDir "train"
$FinalCkpt = Join-Path $TrainDir "coop_lora_siglip\model.pth.tar-$maxEpoch"
$epochs = @(5, 10)

if (-not (Test-Path -LiteralPath $FinalCkpt)) {
    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $seed `
        --trainer CoOpLoRASigLIP `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $TrainDir `
        DATASET.NUM_SHOTS -1 `
        TEST.NO_TEST True `
        OPTIM.MAX_EPOCH $maxEpoch `
        TRAIN.CHECKPOINT_FREQ 5 `
        TRAIN.PRINT_FREQ 100 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TRAIN_X.BATCH_SIZE 64 `
        DATALOADER.TEST.BATCH_SIZE 512
} else {
    "Skipping train because checkpoint exists: $FinalCkpt"
}

foreach ($epoch in $epochs) {
    $Ckpt = Join-Path $TrainDir "coop_lora_siglip\model.pth.tar-$epoch"
    if (-not (Test-Path -LiteralPath $Ckpt)) {
        throw "Missing checkpoint for epoch $epoch`: $Ckpt"
    }

    $EvalDir = Join-Path $ShotDir ("eval_ood_val_epoch" + $epoch)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (-not (Test-Path -LiteralPath $EvalResult)) {
        python -u "$CoOpRoot\train.py" `
            --root "$RepoRoot\data" `
            --seed $seed `
            --trainer CoOpLoRASigLIP `
            --dataset-config-file $DatasetConfig `
            --config-file $TrainerConfig `
            --output-dir $EvalDir `
            --eval-only `
            --model-dir $TrainDir `
            --load-epoch $epoch `
            DATASET.NUM_SHOTS -1 `
            TEST.SPLIT test `
            DATASET.IWILDCAM_TEST_SPLIT val `
            DATALOADER.NUM_WORKERS 4 `
            DATALOADER.TEST.BATCH_SIZE 512
    } else {
        "Skipping ood_val epoch $epoch because result exists: $EvalResult"
    }
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed

$Best = Import-Csv -LiteralPath $SummaryCsv |
    Where-Object { $_.split -like "ood_val_epoch*" } |
    Sort-Object {[double]$_.macro_f1_present} -Descending |
    Select-Object -First 1

if (-not $Best) {
    throw "No ood_val checkpoint rows found in $SummaryCsv"
}

$BestEpoch = [regex]::Match($Best.split, "epoch(\d+)").Groups[1].Value
if (-not $BestEpoch) {
    throw "Could not parse best epoch from split '$($Best.split)'"
}

"Selected epoch $BestEpoch by ood_val macro_f1_present=$($Best.macro_f1_present)"

$splits = @(
    @{Name = "id_val_epoch$BestEpoch"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "id_test_epoch$BestEpoch"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_test_epoch$BestEpoch"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

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
        --trainer CoOpLoRASigLIP `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $EvalDir `
        --eval-only `
        --model-dir $TrainDir `
        --load-epoch $BestEpoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT $split.TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $split.IWildCamTestSplit `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
