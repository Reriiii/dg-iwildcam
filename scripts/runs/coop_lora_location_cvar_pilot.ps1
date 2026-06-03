param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_location_cvar_pilot"
$DetailCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_location_cvar_pilot_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$maxEpoch = 3
$variants = @(
    @{Name = "cvar0p2_rho0p2"; Weight = 0.2; Rho = 0.2},
    @{Name = "cvar0p3_rho0p2"; Weight = 0.3; Rho = 0.2},
    @{Name = "cvar0p5_rho0p3"; Weight = 0.5; Rho = 0.3}
)
$splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"}
)

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing Location-CVaR pilot output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $DetailCsv) {
        "Removing existing Location-CVaR pilot summary: $DetailCsv"
        Remove-Item -LiteralPath $DetailCsv -Force
    }
}

function Invoke-TrainVariant {
    param([hashtable]$Variant)

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$seed\train")
    $FinalCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"
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
        DATASET.NUM_SHOTS -1 `
        TEST.NO_TEST True `
        OPTIM.MAX_EPOCH $maxEpoch `
        TRAIN.CHECKPOINT_FREQ $maxEpoch `
        TRAIN.PRINT_FREQ 100 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TRAIN_X.BATCH_SIZE 64 `
        DATALOADER.TRAIN_X.SAMPLER DomainBalancedSampler `
        DATALOADER.TRAIN_X.N_DOMAIN 8 `
        DATALOADER.TEST.BATCH_SIZE 512 `
        TRAINER.COOPLORA.DOMAIN_CVAR_WEIGHT $Variant.Weight `
        TRAINER.COOPLORA.DOMAIN_CVAR_RHO $Variant.Rho

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
        --load-epoch $maxEpoch `
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

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $DetailCsv --seed $seed
