param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_autoft_loss_probe"
$DetailCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_autoft_loss_probe_detail.csv"
$AggregateCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_autoft_loss_probe_aggregate.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$maxEpoch = 10

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing 16-shot AutoFT-style loss probe output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    foreach ($csv in @($DetailCsv, $AggregateCsv)) {
        if (Test-Path -LiteralPath $csv) {
            "Removing existing 16-shot AutoFT-style loss probe summary: $csv"
            Remove-Item -LiteralPath $csv -Force
        }
    }
}

$AutoFTNoFlypLosses = @(
    "TRAINER.COOPLORA.DCM_WEIGHT", "0.6223533993940668",
    "TRAINER.COOPLORA.ENTROPY_WEIGHT", "0.0004681301224610",
    "TRAINER.COOPLORA.HINGE_WEIGHT", "0.00015961210291135",
    "TRAINER.COOPLORA.L1_INIT_WEIGHT", "0.00154953790283812",
    "TRAINER.COOPLORA.L1_ZERO_WEIGHT", "0.00019307479126688",
    "TRAINER.COOPLORA.L2_INIT_WEIGHT", "0.0001726603186429",
    "TRAINER.COOPLORA.L2_ZERO_WEIGHT", "0.0001090242975318"
)

$Variants = @(
    @{Name = "ce"; Extra = @()},
    @{Name = "dcm0p3"; Extra = @("TRAINER.COOPLORA.DCM_WEIGHT", "0.3")},
    @{Name = "dcm0p6"; Extra = @("TRAINER.COOPLORA.DCM_WEIGHT", "0.6")},
    @{Name = "autoft_no_flyp_sgd"; Extra = $AutoFTNoFlypLosses}
)

$Splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"},
    @{Name = "id_test"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_test"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

function Invoke-VariantTrain {
    param([hashtable]$Variant)

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$seed\train")
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
        [hashtable]$Split
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$seed\train")
    $EvalDir = Join-Path $OutputRoot ("seed_$seed\shot_$shots\eval_" + $Split.Name + "_" + $Variant.Name)
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
        --load-epoch $maxEpoch `
        DATASET.NUM_SHOTS $shots `
        TEST.SPLIT $Split.TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $Split.IWildCamTestSplit `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

foreach ($variant in $Variants) {
    Invoke-VariantTrain -Variant $variant
    foreach ($split in $Splits) {
        Invoke-VariantEval -Variant $variant -Split $split
    }
}

python "$RepoRoot\scripts\tools\summarize_iwildcam_multiseed.py" `
    --root $OutputRoot `
    --output $DetailCsv `
    --aggregate-output $AggregateCsv
