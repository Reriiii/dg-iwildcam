param(
    [switch]$KeepExisting,
    [int]$NumShots = 16,
    [int]$MaxEpoch = 1,
    [double]$Weight = 0.3,
    [double]$Rho = 0.2,
    [int]$NumDomains = 8
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\_smoke_coop_lora_location_cvar"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing Location-CVaR smoke output: $OutputRoot"
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
    DATASET.NUM_SHOTS $NumShots `
    TEST.NO_TEST True `
    OPTIM.MAX_EPOCH $MaxEpoch `
    TRAIN.CHECKPOINT_FREQ $MaxEpoch `
    TRAIN.PRINT_FREQ 20 `
    DATALOADER.NUM_WORKERS 4 `
    DATALOADER.TRAIN_X.BATCH_SIZE 64 `
    DATALOADER.TRAIN_X.SAMPLER DomainBalancedSampler `
    DATALOADER.TRAIN_X.N_DOMAIN $NumDomains `
    DATALOADER.TEST.BATCH_SIZE 512 `
    TRAINER.COOPLORA.DOMAIN_CVAR_WEIGHT $Weight `
    TRAINER.COOPLORA.DOMAIN_CVAR_RHO $Rho
