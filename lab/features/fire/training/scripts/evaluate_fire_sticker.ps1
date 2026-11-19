[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [string]$Root = $PSScriptRoot,
    [string]$DatasetManifest
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
if (Test-Path -LiteralPath (Join-Path $Root 'scripts/common/engine')) {
    $entry = Join-Path $Root 'scripts/train.py'
    $runner = Join-Path $Root 'scripts/run_training_tool.ps1'
    $workspace = $null
    if (-not $DatasetManifest) { $DatasetManifest = 'dataset/workspace_sources.json' }
} else {
    if ($Root -eq $PSScriptRoot) { $Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../../../..')).Path }
    $entry = Join-Path $Root 'lab/features/fire/training/scripts/train_fire_sticker_v4.py'
    $runner = Join-Path $Root 'lab/training_common/run_training_tool.ps1'
    $workspace = $Root
    if (-not $DatasetManifest) { $DatasetManifest = 'datasets/exports/fire/workspace_sources.json' }
}
$resolvedRun = [IO.Path]::GetFullPath((Join-Path $Root $Run))
if (-not $resolvedRun.StartsWith($Root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run must stay inside the workspace or package.'
}
$artifacts = Get-Content -LiteralPath (Join-Path $resolvedRun 'metadata/artifacts.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $artifacts.best) { throw 'The run has no validation-selected checkpoint.' }
$weights = Join-Path $Run $artifacts.best
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'evaluate', '--root', $Root, '--run-dir', $Run, '--weights', $weights,
    '--split', 'val', '--evaluation-name', 'val_selection'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'fire-report', '--root', $Root, '--evaluation', (Join-Path $Run 'evaluation/val_selection'), '--dataset-manifest', $DatasetManifest
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'freeze', '--root', $Root, '--run-dir', $Run, '--weights', $weights,
    '--evaluation-name', 'val_selection', '--selection-reason', 'Best checkpoint selected by full Val only'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'evaluate', '--root', $Root, '--run-dir', $Run, '--weights', (Join-Path $Run 'frozen_selection/selected.pt'),
    '--split', 'test', '--evaluation-name', 'test'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -Workspace $workspace -ScriptArguments @(
    'fire-report', '--root', $Root, '--evaluation', (Join-Path $Run 'evaluation/test'), '--dataset-manifest', $DatasetManifest
)
exit $LASTEXITCODE
