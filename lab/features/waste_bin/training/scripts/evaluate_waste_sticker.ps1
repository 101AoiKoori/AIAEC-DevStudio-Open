[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Run,
    [string]$Root = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
$artifactsPath = Join-Path $Root (Join-Path $Run 'metadata/artifacts.json')
$artifacts = Get-Content -LiteralPath $artifactsPath -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $artifacts.best) { throw 'The run has no validation-selected checkpoint.' }
$weights = Join-Path $Run $artifacts.best
$entry = Join-Path $Root 'scripts/train.py'
$runner = Join-Path $Root 'scripts/run_training_tool.ps1'
$validationName = 'val_selection'

& $runner -Entry $entry -ScriptArguments @(
    'evaluate', '--run-dir', $Run, '--weights', $weights,
    '--split', 'val', '--evaluation-name', $validationName
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -ScriptArguments @(
    'freeze', '--run-dir', $Run, '--weights', $weights,
    '--evaluation-name', $validationName, '--selection-reason', 'Best checkpoint selected by full validation'
)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $runner -Entry $entry -ScriptArguments @(
    'evaluate', '--run-dir', $Run, '--weights', (Join-Path $Run 'frozen_selection/selected.pt'),
    '--split', 'test', '--evaluation-name', 'test'
)
exit $LASTEXITCODE
