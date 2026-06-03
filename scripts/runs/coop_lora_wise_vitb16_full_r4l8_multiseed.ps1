param(
    [switch]$KeepExisting,
    [int[]]$Seeds = @(1, 2, 3, 4, 5)
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed"
$DetailCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_detail.csv"
$AggregateCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_aggregate.csv"
$SelectedIdCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_selected_idval_detail.csv"
$SelectedIdAggregateCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_selected_idval_aggregate.csv"
$SelectedOodCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_selected_oodval_detail.csv"
$SelectedOodAggregateCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_selected_oodval_aggregate.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$maxEpoch = 10
$wiseEpoch = 1
$ExistingSeed1Source = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_ablation\r4_l8\train\coop_lora\model.pth.tar-10"
$alphas = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.78, 0.8, 0.82, 0.85, 0.9, 1.0)

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing CoOpLoRA-WiSE multiseed output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    foreach ($csv in @($DetailCsv, $AggregateCsv, $SelectedIdCsv, $SelectedIdAggregateCsv, $SelectedOodCsv, $SelectedOodAggregateCsv)) {
        if (Test-Path -LiteralPath $csv) {
            "Removing existing CoOpLoRA-WiSE multiseed summary: $csv"
            Remove-Item -LiteralPath $csv -Force
        }
    }
}

function Format-AlphaName {
    param([double]$Alpha)
    if ([Math]::Abs($Alpha - [Math]::Round($Alpha, 1)) -lt 1e-9) {
        return ("a{0:0.0}" -f $Alpha).Replace(".", "p")
    }
    return ("a{0:0.##}" -f $Alpha).Replace(".", "p")
}

function Get-SourceCheckpoint {
    param([int]$Seed)

    if ($Seed -eq 1 -and (Test-Path -LiteralPath $ExistingSeed1Source)) {
        return $ExistingSeed1Source
    }

    $TrainDir = Join-Path $OutputRoot "models\source\seed_$Seed\train"
    return (Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch")
}

function Get-SourceTrainDir {
    param([int]$Seed)
    return (Join-Path $OutputRoot "models\source\seed_$Seed\train")
}

function Invoke-SourceTrain {
    param([int]$Seed)

    $SourceCkpt = Get-SourceCheckpoint -Seed $Seed
    if (Test-Path -LiteralPath $SourceCkpt) {
        "Skipping source train seed $Seed because checkpoint exists: $SourceCkpt"
        return
    }

    $TrainDir = Get-SourceTrainDir -Seed $Seed
    python -u "$CoOpRoot\train.py" `
        --root "$RepoRoot\data" `
        --seed $Seed `
        --trainer CoOpLoRA `
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
        DATALOADER.TRAIN_X.SAMPLER RandomSampler `
        DATALOADER.TEST.BATCH_SIZE 512

    if (-not (Test-Path -LiteralPath $SourceCkpt)) {
        throw "Missing trained source checkpoint for seed ${Seed}: $SourceCkpt"
    }
}

function Invoke-CoOpLoRAEval {
    param(
        [int]$Seed,
        [string]$EvalName,
        [string]$ModelDir,
        [int]$LoadEpoch,
        [string]$TestSplit,
        [string]$IWildCamTestSplit
    )

    $ShotDir = Join-Path $OutputRoot "seed_$Seed\shot_full"
    $EvalDir = Join-Path $ShotDir ("eval_" + $EvalName)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $EvalName seed $Seed because result exists: $EvalResult"
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
        --model-dir $ModelDir `
        --load-epoch $LoadEpoch `
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT $TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $IWildCamTestSplit `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

foreach ($seed in $Seeds) {
    Invoke-SourceTrain -Seed $seed
    $SourceCkpt = Get-SourceCheckpoint -Seed $seed

    foreach ($alpha in $alphas) {
        $Name = Format-AlphaName $alpha
        $ModelDir = Join-Path $OutputRoot "models\wise\seed_$seed\$Name\train"
        $WiseCkpt = Join-Path $ModelDir "coop_lora\model.pth.tar-$wiseEpoch"

        if (-not (Test-Path -LiteralPath $WiseCkpt)) {
            python "$RepoRoot\scripts\tools\make_lora_wise.py" `
                --source $SourceCkpt `
                --output $WiseCkpt `
                --alpha $alpha `
                --ctx-init "a camera trap photo of" `
                --backbone "ViT-B/16" `
                --coop-root $CoOpRoot `
                --epoch $wiseEpoch
        } else {
            "Skipping WiSE checkpoint creation seed $seed $Name because checkpoint exists: $WiseCkpt"
        }

        Invoke-CoOpLoRAEval -Seed $seed -EvalName "id_val_$Name" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "val" -IWildCamTestSplit "val"
        Invoke-CoOpLoRAEval -Seed $seed -EvalName "ood_val_$Name" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "val"
    }
}

python "$RepoRoot\scripts\tools\summarize_iwildcam_multiseed.py" `
    --root $OutputRoot `
    --output $DetailCsv `
    --aggregate-output $AggregateCsv

$DetailRows = Import-Csv -LiteralPath $DetailCsv
foreach ($seed in $Seeds) {
    $BestId = $DetailRows |
        Where-Object { [int]$_.seed -eq $seed -and $_.split -eq "id_val" -and $_.variant -like "a*" } |
        Sort-Object {[double]$_.accuracy} -Descending |
        Select-Object -First 1
    $BestOod = $DetailRows |
        Where-Object { [int]$_.seed -eq $seed -and $_.split -eq "ood_val" -and $_.variant -like "a*" } |
        Sort-Object {[double]$_.macro_f1_present} -Descending |
        Select-Object -First 1

    if (-not $BestId) {
        throw "No ID-val alpha found for seed $seed"
    }
    if (-not $BestOod) {
        throw "No OOD-val alpha found for seed $seed"
    }

    "Seed $seed selected ID-val alpha=$($BestId.variant) by accuracy=$($BestId.accuracy)"
    "Seed $seed selected OOD-val alpha=$($BestOod.variant) by macro_f1_present=$($BestOod.macro_f1_present)"

    $SelectedCases = @($BestId.variant, $BestOod.variant) | Select-Object -Unique
    foreach ($case in $SelectedCases) {
        $ModelDir = Join-Path $OutputRoot "models\wise\seed_$seed\$case\train"
        Invoke-CoOpLoRAEval -Seed $seed -EvalName "id_test_$case" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "id_test"
        Invoke-CoOpLoRAEval -Seed $seed -EvalName "ood_test_$case" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "test"
    }
}

python "$RepoRoot\scripts\tools\summarize_iwildcam_multiseed.py" `
    --root $OutputRoot `
    --output $DetailCsv `
    --aggregate-output $AggregateCsv

python "$RepoRoot\scripts\tools\summarize_wise_selection.py" `
    --detail $DetailCsv `
    --output $SelectedIdCsv `
    --aggregate-output $SelectedIdAggregateCsv `
    --selection-name "id_val_accuracy" `
    --selection-split "id_val" `
    --selection-metric "accuracy"

python "$RepoRoot\scripts\tools\summarize_wise_selection.py" `
    --detail $DetailCsv `
    --output $SelectedOodCsv `
    --aggregate-output $SelectedOodAggregateCsv `
    --selection-name "ood_val_macro_f1" `
    --selection-split "ood_val" `
    --selection-metric "macro_f1_present"
