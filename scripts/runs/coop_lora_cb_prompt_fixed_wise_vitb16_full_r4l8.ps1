param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8_cb.yaml"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_cb_pfwise"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_cb_pfwise_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

$seed = 1
$maxEpoch = 10
$wiseEpoch = 1
$ShotDir = Join-Path $OutputRoot "shot_full"
$TrainDir = Join-Path $ShotDir "train"
$SourceCkpt = Join-Path $TrainDir "coop_lora\model.pth.tar-$maxEpoch"
$alphas = @(0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.88, 0.9, 0.95, 1.0)

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing output for clean training: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing summary for clean training: $SummaryCsv"
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
        DATASET.NUM_SHOTS -1 `
        TEST.SPLIT $TestSplit `
        DATASET.IWILDCAM_TEST_SPLIT $IWildCamTestSplit `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TEST.BATCH_SIZE 512
}

if (-not (Test-Path -LiteralPath $SourceCkpt)) {
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
        TRAIN.CHECKPOINT_FREQ 5 `
        TRAIN.PRINT_FREQ 100 `
        DATALOADER.NUM_WORKERS 4 `
        DATALOADER.TRAIN_X.BATCH_SIZE 64 `
        DATALOADER.TRAIN_X.SAMPLER ClassBalancedSampler `
        DATALOADER.TRAIN_X.CLASS_BALANCED_POWER 0.5 `
        DATALOADER.TEST.BATCH_SIZE 512
} else {
    "Skipping train because checkpoint exists: $SourceCkpt"
}

if (-not (Test-Path -LiteralPath $SourceCkpt)) {
    throw "Missing trained checkpoint: $SourceCkpt"
}

foreach ($alpha in $alphas) {
    $Name = Format-AlphaName $alpha
    $ModelDir = Join-Path $OutputRoot ("models\pf_" + $Name + "\train")
    $WiseCkpt = Join-Path $ModelDir "coop_lora\model.pth.tar-$wiseEpoch"

    if (-not (Test-Path -LiteralPath $WiseCkpt)) {
        python "$RepoRoot\scripts\tools\make_lora_wise.py" `
            --source $SourceCkpt `
            --output $WiseCkpt `
            --alpha $alpha `
            --prompt-alpha 1.0 `
            --lora-alpha $alpha `
            --ctx-init "a camera trap photo of" `
            --backbone "ViT-B/16" `
            --coop-root $CoOpRoot `
            --epoch $wiseEpoch
    } else {
        "Skipping prompt-fixed WiSE checkpoint creation for pf_$Name because checkpoint exists: $WiseCkpt"
    }

    Invoke-CoOpLoRAEval -EvalName "id_val_pf_$Name" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "val" -IWildCamTestSplit "val"
    Invoke-CoOpLoRAEval -EvalName "ood_val_pf_$Name" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "val"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed

$BestId = Import-Csv -LiteralPath $SummaryCsv |
    Where-Object { $_.split -like "id_val_pf_a*" } |
    Sort-Object {[double]$_.accuracy} -Descending |
    Select-Object -First 1

$BestOod = Import-Csv -LiteralPath $SummaryCsv |
    Where-Object { $_.split -like "ood_val_pf_a*" } |
    Sort-Object {[double]$_.macro_f1_present} -Descending |
    Select-Object -First 1

if (-not $BestId) {
    throw "No id_val prompt-fixed WiSE rows found in $SummaryCsv"
}
if (-not $BestOod) {
    throw "No ood_val prompt-fixed WiSE rows found in $SummaryCsv"
}

$BestIdCase = $BestId.split.Replace("id_val_", "")
$BestOodCase = $BestOod.split.Replace("ood_val_", "")
"Selected ID-val case=$BestIdCase by accuracy=$($BestId.accuracy)"
"Selected OOD-val case=$BestOodCase by macro_f1_present=$($BestOod.macro_f1_present)"

$SelectedCases = @($BestIdCase, $BestOodCase, "pf_a1p0") | Select-Object -Unique
foreach ($case in $SelectedCases) {
    $ModelDir = Join-Path $OutputRoot ("models\" + $case + "\train")
    Invoke-CoOpLoRAEval -EvalName "id_test_$case" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "id_test"
    Invoke-CoOpLoRAEval -EvalName "ood_test_$case" -ModelDir $ModelDir -LoadEpoch $wiseEpoch -TestSplit "test" -IWildCamTestSplit "test"
}

python "$RepoRoot\scripts\tools\summarize_iwildcam.py" --root $OutputRoot --output $SummaryCsv --seed $seed
