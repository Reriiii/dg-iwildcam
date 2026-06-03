param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$WiseRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_logit_ensemble"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_logit_ensemble_summary.csv"
$IdSelectedCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_selected_idval_detail.csv"
$OodSelectedCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise_multiseed_selected_oodval_detail.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing logit ensemble output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing logit ensemble summary: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
}

function Get-ModelArgs {
    param([string]$SelectionCsv)

    $rows = Import-Csv -LiteralPath $SelectionCsv
    $args = @()
    foreach ($row in $rows) {
        $modelDir = Join-Path $WiseRoot ("models\wise\seed_$($row.seed)\$($row.variant)\train")
        $ckpt = Join-Path $modelDir "coop_lora\model.pth.tar-1"
        if (-not (Test-Path -LiteralPath $ckpt)) {
            throw "Missing ensemble checkpoint: $ckpt"
        }
        $args += @("--model", $modelDir)
    }
    return $args
}

function Invoke-EnsembleEval {
    param(
        [string]$EnsembleName,
        [string]$SelectionCsv,
        [string]$SplitName,
        [string]$TestSplit,
        [string]$IWildCamTestSplit
    )

    $evalDir = Join-Path $OutputRoot ("eval_" + $SplitName + "_" + $EnsembleName)
    $evalSummary = Join-Path $evalDir "summary.csv"
    if (Test-Path -LiteralPath $evalSummary) {
        "Skipping ensemble eval $SplitName $EnsembleName because summary exists: $evalSummary"
        return
    }

    $modelArgs = Get-ModelArgs -SelectionCsv $SelectionCsv
    python -u "$RepoRoot\scripts\tools\eval_coop_lora_ensemble.py" `
        --repo-root $RepoRoot `
        --root "$RepoRoot\data" `
        --dataset-config-file $DatasetConfig `
        --config-file $TrainerConfig `
        --output-dir $evalDir `
        --ensemble-name $EnsembleName `
        --split-name $SplitName `
        --test-split $TestSplit `
        --iwildcam-test-split $IWildCamTestSplit `
        --num-shots -1 `
        --seed 1 `
        --batch-size 512 `
        --num-workers 4 `
        @modelArgs
}

$jobs = @(
    @{Ensemble = "idval_selected"; Csv = $IdSelectedCsv},
    @{Ensemble = "oodval_selected"; Csv = $OodSelectedCsv}
)
$splits = @(
    @{Name = "id_val"; TestSplit = "val"; IWildCamTestSplit = "val"},
    @{Name = "ood_val"; TestSplit = "test"; IWildCamTestSplit = "val"},
    @{Name = "id_test"; TestSplit = "test"; IWildCamTestSplit = "id_test"},
    @{Name = "ood_test"; TestSplit = "test"; IWildCamTestSplit = "test"}
)

foreach ($job in $jobs) {
    foreach ($split in $splits) {
        Invoke-EnsembleEval `
            -EnsembleName $job.Ensemble `
            -SelectionCsv $job.Csv `
            -SplitName $split.Name `
            -TestSplit $split.TestSplit `
            -IWildCamTestSplit $split.IWildCamTestSplit
    }
}

$rows = @()
foreach ($summary in Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter "summary.csv") {
    $rows += Import-Csv -LiteralPath $summary.FullName
}
$rows | Sort-Object split, ensemble | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation
$rows | Sort-Object split, ensemble | Format-Table -AutoSize
"Saved ensemble summary: $SummaryCsv"
