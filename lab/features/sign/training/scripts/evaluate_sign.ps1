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
    $entry = Join-Path $Root 'lab/features/sign/training/scripts/train_sign_v4.py'
    $runner = Join-Path $Root 'lab/training_common/run_training_tool.ps1'
    $workspace = $Root
    if (-not $DatasetManifest) { $DatasetManifest = 'datasets/exports/sign/workspace_sources.json' }
}
$artifacts = Get-Content -LiteralPath (Join-Path $Root (Join-Path $Run 'metadata/artifacts.json')) -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $artifacts.best) { throw 'The run has no validation-selected checkpoint.' }
$weights = Join-Path $Run $artifacts.best
$validationName = 'val_selection'
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'evaluate', '--root', $Root, '--run-dir', $Run, '--weights', $weights,
    '--split', 'val', '--evaluation-name', $validationName
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'sign-report', '--root', $Root, '--evaluation', (Join-Path $Run "evaluation/$validationName"),
    '--dataset-manifest', $DatasetManifest
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'freeze', '--root', $Root, '--run-dir', $Run, '--weights', $weights,
    '--evaluation-name', $validationName, '--selection-reason', 'Best checkpoint selected by validation only'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'evaluate', '--root', $Root, '--run-dir', $Run, '--weights', (Join-Path $Run 'frozen_selection/selected.pt'),
    '--split', 'test', '--evaluation-name', 'test'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'sign-report', '--root', $Root, '--evaluation', (Join-Path $Run 'evaluation/test'),
    '--dataset-manifest', $DatasetManifest
)
exit $LASTEXITCODE
