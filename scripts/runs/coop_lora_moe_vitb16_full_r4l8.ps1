param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Workplace\DG-iWildCam"
$CoOpRoot = Join-Path $RepoRoot "external\CoOp"
$DasslRoot = Join-Path $RepoRoot "external\Dassl.pytorch"
$DatasetConfig = Join-Path $CoOpRoot "configs\datasets\iwildcam.yaml"
$TrainerConfig = Join-Path $CoOpRoot "configs\trainers\CoOpLoRA\vit_b16_iwildcam_full_10ep_r4_l8.yaml"
$GeneralistModel = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_wise\models\a0p8\train"
$SpecialistModel = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_10ep_r4_l8_cb_pfwise\models\pf_a1p0\train"
$OutputRoot = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_moe_r4_l8"
$SummaryCsv = Join-Path $RepoRoot "outputs\coop_lora_iwildcam_vitb16_full_moe_r4_l8_summary.csv"

$env:PYTHONPATH = "$DasslRoot;$CoOpRoot;$env:PYTHONPATH"

if (-not $KeepExisting) {
    if (Test-Path -LiteralPath $OutputRoot) {
        "Removing existing MoE output: $OutputRoot"
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    if (Test-Path -LiteralPath $SummaryCsv) {
        "Removing existing MoE summary: $SummaryCsv"
        Remove-Item -LiteralPath $SummaryCsv -Force
    }
}

function Invoke-MoEEval {
    param(
        [string]$SplitName,
        [string]$TestSplit,
        [string]$IWildCamTestSplit,
        [string[]]$OnlyCandidateNames = @(),
        [switch]$WritePerClass
    )

    $EvalDir = Join-Path $OutputRoot ("eval_" + $SplitName)
    $Summary = Join-Path $EvalDir "summary.csv"
    if (Test-Path -LiteralPath $Summary) {
        "Skipping MoE eval $SplitName because summary exists: $Summary"
        return
    }

    $PythonArgs = @(
        "$RepoRoot\scripts\tools\eval_coop_lora_moe.py",
        "--repo-root", $RepoRoot,
        "--coop-root", $CoOpRoot,
        "--dassl-root", $DasslRoot,
        "--root", "$RepoRoot\data",
        "--dataset-config-file", $DatasetConfig,
        "--config-file", $TrainerConfig,
        "--output-dir", $EvalDir,
        "--generalist-model", $GeneralistModel,
        "--specialist-model", $SpecialistModel,
        "--generalist-epoch", "1",
        "--specialist-epoch", "1",
        "--split-name", $SplitName,
        "--test-split", $TestSplit,
        "--iwildcam-test-split", $IWildCamTestSplit,
        "--num-shots", "-1",
        "--seed", "1",
        "--batch-size", "512",
        "--num-workers", "4",
        "--grid", "global",
        "--grid", "bucket"
    )

    foreach ($Name in $OnlyCandidateNames) {
        $PythonArgs += @("--only-candidate-name", $Name)
    }
    if ($WritePerClass) {
        $PythonArgs += "--write-per-class"
    }

    python -u @PythonArgs
}

Invoke-MoEEval -SplitName "id_val" -TestSplit "val" -IWildCamTestSplit "val"
Invoke-MoEEval -SplitName "ood_val" -TestSplit "test" -IWildCamTestSplit "val"

$IdValSummary = Join-Path $OutputRoot "eval_id_val\summary.csv"
$OodValSummary = Join-Path $OutputRoot "eval_ood_val\summary.csv"

$BestId = Import-Csv -LiteralPath $IdValSummary |
    Sort-Object -Property @{Expression = {[double]$_.accuracy}; Descending = $true}, @{Expression = {[double]$_.macro_f1_present}; Descending = $true} |
    Select-Object -First 1
$BestOod = Import-Csv -LiteralPath $OodValSummary |
    Sort-Object -Property @{Expression = {[double]$_.macro_f1_present}; Descending = $true}, @{Expression = {[double]$_.accuracy}; Descending = $true} |
    Select-Object -First 1

if (-not $BestId) {
    throw "No ID-val MoE candidate found in $IdValSummary"
}
if (-not $BestOod) {
    throw "No OOD-val MoE candidate found in $OodValSummary"
}

"Selected ID-val MoE candidate=$($BestId.candidate), accuracy=$($BestId.accuracy), macro_f1=$($BestId.macro_f1_present)"
"Selected OOD-val MoE candidate=$($BestOod.candidate), accuracy=$($BestOod.accuracy), macro_f1=$($BestOod.macro_f1_present)"

$SelectedCandidates = @($BestId.candidate, $BestOod.candidate, "g0", "g1") | Select-Object -Unique
Invoke-MoEEval -SplitName "id_test" -TestSplit "test" -IWildCamTestSplit "id_test" -OnlyCandidateNames $SelectedCandidates -WritePerClass
Invoke-MoEEval -SplitName "ood_test" -TestSplit "test" -IWildCamTestSplit "test" -OnlyCandidateNames $SelectedCandidates -WritePerClass

$AllRows = @()
foreach ($Csv in @(
    (Join-Path $OutputRoot "eval_id_val\summary.csv"),
    (Join-Path $OutputRoot "eval_ood_val\summary.csv"),
    (Join-Path $OutputRoot "eval_id_test\summary.csv"),
    (Join-Path $OutputRoot "eval_ood_test\summary.csv")
)) {
    if (Test-Path -LiteralPath $Csv) {
        $AllRows += Import-Csv -LiteralPath $Csv
    }
}

$AllRows | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation
"Saved MoE summary: $SummaryCsv"
