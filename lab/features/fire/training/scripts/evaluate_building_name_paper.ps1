[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [string]$Root = $PSScriptRoot,
    [string]$DatasetManifest
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
if (Test-Path -LiteralPath (Join-Path $Root 'scripts/train.py') -PathType Leaf) {
    $entry = Join-Path $Root 'scripts/train.py'
    $runner = Join-Path $Root 'scripts/run_training_tool.ps1'
    $workspace = $null
    if (-not $DatasetManifest) { $DatasetManifest = 'dataset/workspace_sources.json' }
} else {
    if ($Root -eq $PSScriptRoot) { $Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../../../..')).Path }
    $entry = Join-Path $Root 'lab/features/fire/training/scripts/train_building_name_paper_v4.py'
    $runner = Join-Path $Root 'lab/training_common/run_training_tool.ps1'
    $workspace = $Root
    if (-not $DatasetManifest) { $DatasetManifest = 'datasets/exports/building_name_paper/workspace_sources.json' }
}
$resolvedRun = [IO.Path]::GetFullPath((Join-Path $Root $Run))
if (-not $resolvedRun.StartsWith($Root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run must stay inside the workspace or package.'
}
$artifacts = Get-Content -LiteralPath (Join-Path $resolvedRun 'metadata/artifacts.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $artifacts.best) { throw 'The run has no validation-selected checkpoint.' }
$weights = Join-Path $Run $artifacts.best
$validationName = 'val_selection'
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'evaluate', '--root', $Root, '--run-dir', $Run, '--weights', $weights,
    '--split', 'val', '--evaluation-name', $validationName
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
foreach ($report in @('paper-report', 'paper-sizes')) {
    $arguments = @($report, '--root', $Root, '--evaluation', (Join-Path $Run "evaluation/$validationName"))
    if ($report -eq 'paper-report') { $arguments += @('--dataset-manifest', $DatasetManifest) }
    & $runner -Entry $entry -Workspace $workspace -ScriptArguments $arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'freeze', '--root', $Root, '--run-dir', $Run, '--weights', $weights,
    '--evaluation-name', $validationName, '--selection-reason', 'Best checkpoint selected by full Val only'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'evaluate', '--root', $Root, '--run-dir', $Run, '--weights', (Join-Path $Run 'frozen_selection/selected.pt'),
    '--split', 'test', '--evaluation-name', 'test'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
foreach ($report in @('paper-report', 'paper-sizes')) {
    $arguments = @($report, '--root', $Root, '--evaluation', (Join-Path $Run 'evaluation/test'))
    if ($report -eq 'paper-report') { $arguments += @('--dataset-manifest', $DatasetManifest) }
    & $runner -Entry $entry -Workspace $workspace -ScriptArguments $arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit 0
