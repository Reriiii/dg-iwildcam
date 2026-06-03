param(
    [switch]$KeepExisting,
    [int[]]$Seeds = @(1, 2, 3)
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_supcon_multiseed"
$DetailCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_supcon_multiseed_detail.csv"
$AggregateCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_supcon_multiseed_aggregate.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$shots = 16
$maxEpoch = 10

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing 16-shot SupCon multi-seed output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    foreach ($csv in @($DetailCsv, $AggregateCsv)) {
        if (Test-Path -LiteralPath $csv) {
            "Removing existing 16-shot SupCon multi-seed summary: $csv"
            Remove-Item -LiteralPath $csv -Force
        }
    }
}

$Variants = @(
    @{Name = "ce"; Extra = @("TRAINER.COOPLORA.SUPCON_WEIGHT", "0.0", "TRAINER.COOPLORA.L2_INIT_WEIGHT", "0.0")},
    @{Name = "supcon0p1"; Extra = @("TRAINER.COOPLORA.SUPCON_WEIGHT", "0.1", "TRAINER.COOPLORA.SUPCON_TEMPERATURE", "1.0", "TRAINER.COOPLORA.L2_INIT_WEIGHT", "0.0")},
    @{Name = "supcon0p5"; Extra = @("TRAINER.COOPLORA.SUPCON_WEIGHT", "0.5", "TRAINER.COOPLORA.SUPCON_TEMPERATURE", "1.0", "TRAINER.COOPLORA.L2_INIT_WEIGHT", "0.0")}
)

$Splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"},
    @{Name = "id_test"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_test"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

function Invoke-VariantTrain {
    param(
        [hashtable]$Variant,
        [int]$Seed
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$Seed\train")
    $FinalCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"
    $ExtraArgs = @($Variant.Extra)
    if (Test-Path -LiteralPath $FinalCkpt) {
        "Skipping train $($Variant.Name) seed $Seed because checkpoint exists: $FinalCkpt"
        return
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $Seed `
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
        throw "Missing trained checkpoint for $($Variant.Name) seed ${Seed}: $FinalCkpt"
    }
}

function Invoke-VariantEval {
    param(
        [hashtable]$Variant,
        [int]$Seed,
        [hashtable]$Split
    )

    $TrainDir = Join-Path $OutputRoot ("models\" + $Variant.Name + "\seed_$Seed\train")
    $EvalDir = Join-Path $OutputRoot ("seed_$Seed\shot_$shots\eval_" + $Split.Name + "_" + $Variant.Name)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    $ExtraArgs = @($Variant.Extra)
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $($Split.Name) $($Variant.Name) seed $Seed because result exists: $EvalResult"
        return
    }

    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $Seed `
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
        DATALOADER.TEST.BATCH_SIZE 512 `
        @ExtraArgs
}

foreach ($seed in $Seeds) {
    foreach ($variant in $Variants) {
        Invoke-VariantTrain -Variant $variant -Seed $seed
        foreach ($split in $Splits) {
            Invoke-VariantEval -Variant $variant -Seed $seed -Split $split
        }
    }
}

python "$RepoRoot\scripts\tools\summarize_iwildcam_multiseed.py" `
    --root $OutputRoot `
    --output $DetailCsv `
    --aggregate-output $AggregateCsv
