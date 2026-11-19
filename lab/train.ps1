[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Task,
    [switch]$Resume,
    [string]$Run,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
if ($Run -and -not $Resume) { throw '-Run requires -Resume.' }
$action = if ($Resume) { 'resume' } else { 'train' }
$launchArgs = @($action)
if ($Run) { $launchArgs += @('--run', $Run) }
if ($Plan) { $launchArgs += '--plan' }

$entry = Join-Path $PSScriptRoot 'training_common\launch_task.py'
$workspace = ''
if (Test-Path -LiteralPath $entry -PathType Leaf) {
    if (-not $Task) { throw 'Specify a V4 task: person, e_bike, fire, building_name_paper, sign, waste_bin_state, waste_sticker, meter.' }
    $launchArgs += @('--task', $Task)
    $workspace = Split-Path -Parent $PSScriptRoot
} else {
    if ($Task) { throw 'This package already selects the task. Use .\train.ps1 or .\train.ps1 -Resume.' }
    $entry = Join-Path $PSScriptRoot 'scripts\launch_task.py'
}
& (Join-Path (Split-Path -Parent $entry) 'run_training_tool.ps1') -Entry $entry -ScriptArguments $launchArgs -Workspace $workspace
exit $LASTEXITCODE
