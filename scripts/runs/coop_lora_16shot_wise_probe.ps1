param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_wise_probe"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_wise_probe_summary.csv"
$SourceTrainDir = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_16shot_ctx5_r4_l8_seed1\shot_16\train"
$SourceCkpt = Join-Path $SourceTrainDir "coop_lora\model.pth.tar-10"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$shots = 16
$sourceEpoch = 10
$wiseEpoch = 1
$ShotDir = Join-Path $OutputRoot "shot_$shots"
$alphas = @(0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)

if (-not (Test-Path -LiteralPath $SourceCkpt)) {
    throw "Missing source 16-shot checkpoint: $SourceCkpt"
}

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing 16-shot WiSE probe output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing 16-shot WiSE probe summary: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
}

function Format-AlphaName {
    param([double]$Alpha)
    if ([Math]::Abs($Alpha - [Math]::Round($Alpha, 1)) -lt 1e-9) {
        return ("a{0:0.0}" -f $Alpha).Replace(".", "p")
    }
    return ("a{0:0.##}" -f $Alpha).Replace(".", "p")
}

function Invoke-CoOpLoRAEval {
    param(
        [string]$EvalName,
        [string]$ModelDir,
        [int]$LoadEpoch,
        [string]$TestSplit,
        [string]$IWildCamTestSplit
    )

    $EvalDir = Join-Path $ShotDir ("eval_" + $EvalName)
    $EvalResult = Join-Path $EvalDir "iwildcam_per_class_eval1.csv"
    if (Test-Path -LiteralPath $EvalResult) {
        "Skipping eval $EvalName because result exists: $EvalResult"
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
        --model-dir $ModelDir `
        --load-epoch $LoadEpoch `
        DATASET.NUM_SHOTS $shots `
        TEST.SPLIT $TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $IWildCamTestSplit `
        TRAINER.COOPLORA.KL_WEIGHT 0.0 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

function New-WiseCheckpoint {
    param(
        [string]$Name,
        [string]$ModelDir,
        [double]$Alpha,
        [double]$PromptAlpha,
        [double]$LoraAlpha
    )

    $WiseCkpt = Join-Path $ModelDir "coop_lora\model.pth.tar-$wiseEpoch"
    if (Test-Path -LiteralPath $WiseCkpt) {
        "Skipping checkpoint creation for $Name because checkpoint exists: $WiseCkpt"
        return
    }

    python "$RepoRoot\scripts\tools\make_lora_wise.py" `
        --source $SourceCkpt `
        --output $WiseCkpt `
        --alpha $Alpha `
        --prompt-alpha $PromptAlpha `
        --lora-alpha $LoraAlpha `
        --ctx-init "a camera trap photo of" `
        --backbone "ViT-B/16" `
        --coop-root $CoOpRoot `
        --epoch $wiseEpoch
}

Invoke-CoOpLoRAEval -EvalName "id_val_single" -ModelDir $SourceTrainDir -LoadEpoch $sourceEpoch -TestSplit "val" -IWildCamTestSplit "val"
Invoke-CoOpLoRAEval -EvalName "ood_val_single" -ModelDir $SourceTrainDir -LoadEpoch $sourceEpoch -TestSplit "test" -IWildCamTestSplit "val"

foreach ($alpha in $alphas) {
    $AlphaName = Format-AlphaName $alpha

    $WiseName = "wise_$AlphaName"
    $WiseModelDir = Join-Path $OutputRoot ("models\" + $WiseName + "\train")
    New-WiseCheckpoint -Name $WiseName -ModelDir $WiseModelDir -Alpha $alpha -PromptAlpha $alpha -LoraAlpha $alpha
    Invoke-CoOpLoRAEval -EvalName "id_val_$WiseName" -ModelDir $WiseModelDir -LoadEpoch $wiseEpoch -TestSplit "val" -IWildCamTestSplit "val"
    Invoke-CoOpLoRAEval -EvalName "ood_val_$WiseName" -ModelDir $WiseModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "val"

    $PromptFixedName = "pf_$AlphaName"
    $PromptFixedModelDir = Join-Path $OutputRoot ("models\" + $PromptFixedName + "\train")
    New-WiseCheckpoint -Name $PromptFixedName -ModelDir $PromptFixedModelDir -Alpha $alpha -PromptAlpha 1.0 -LoraAlpha $alpha
    Invoke-CoOpLoRAEval -EvalName "id_val_$PromptFixedName" -ModelDir $PromptFixedModelDir -LoadEpoch $wiseEpoch -TestSplit "val" -IWildCamTestSplit "val"
    Invoke-CoOpLoRAEval -EvalName "ood_val_$PromptFixedName" -ModelDir $PromptFixedModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "val"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed

$Rows = Import-Csv -LiteralPath $SummaryCsv
$BestOodCases = $Rows |
    Where-Object { $_.split -like "ood_val_*" -and $_.split -ne "ood_val_single" } |
    Sort-Object {[double]$_.macro_f1_present} -Descending |
    Select-Object -First 3

$SelectedCases = @("single")
$SelectedCases += $BestOodCases | ForEach-Object { $_.split.Replace("ood_val_", "") }
$SelectedCases = $SelectedCases | Select-Object -Unique

foreach ($case in $SelectedCases) {
    if ($case -eq "single") {
        $ModelDir = $SourceTrainDir
        $LoadEpoch = $sourceEpoch
    } else {
        $ModelDir = Join-Path $OutputRoot ("models\" + $case + "\train")
        $LoadEpoch = $wiseEpoch
    }

    Invoke-CoOpLoRAEval -EvalName "id_test_$case" -ModelDir $ModelDir -LoadEpoch $LoadEpoch -TestSplit "test" -IWildCamTestSplit "id_test"
    Invoke-CoOpLoRAEval -EvalName "ood_test_$case" -ModelDir $ModelDir -LoadEpoch $LoadEpoch -TestSplit "test" -IWildCamTestSplit "test"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
