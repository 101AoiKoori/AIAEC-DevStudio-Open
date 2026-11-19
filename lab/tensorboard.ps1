[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Task,
    [string]$Run,
    [switch]$Package,
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$workspace = ''
$boardArgs = @()
$entry = Join-Path $PSScriptRoot 'training_common\tensorboard_v3.py'
if (Test-Path -LiteralPath $entry -PathType Leaf) {
    if (-not $Task) { throw 'Specify a V4 task: person, e_bike, fire, building_name_paper, sign, waste_bin_state, waste_sticker, meter.' }
    $workspace = Split-Path -Parent $PSScriptRoot
    $boardArgs += @('--task', $Task)
} else {
    if ($Task -or $Package) { throw 'This package already selects the task. Use .\tensorboard.ps1.' }
    $entry = Join-Path $PSScriptRoot 'scripts\view_tensorboard.py'
}
if ($Package) { $boardArgs += '--package' }
if ($Run) { $boardArgs += @('--run', $Run) }
if ($Plan) { $boardArgs += '--plan' }
& (Join-Path (Split-Path -Parent $entry) 'run_training_tool.ps1') -Entry $entry -ScriptArguments $boardArgs -Workspace $workspace -Portable:$Package
exit $LASTEXITCODE
