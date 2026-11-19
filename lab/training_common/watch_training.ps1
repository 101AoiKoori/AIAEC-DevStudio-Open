[CmdletBinding()]
param(
    [string]$Environment = '',
    [string]$Python = '',
    [string]$RunDir = '',
    [ValidateRange(0.1,3600)][double]$IntervalSeconds = 2,
    [switch]$Follow
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$packageRoot = Split-Path -Parent $PSScriptRoot
$monitor = Join-Path $PSScriptRoot 'monitor_training.py'
if (-not (Test-Path -LiteralPath $monitor -PathType Leaf)) {
    throw "Training monitor missing: $monitor"
}
if (-not $Environment) {
    foreach ($candidate in @((Join-Path $packageRoot 'portable_env'),
                              (Join-Path (Split-Path -Parent $packageRoot) 'portable_env'))) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'run.ps1') -PathType Leaf) {
            $Environment = $candidate
            break
        }
    }
}
if (-not $Environment) {
    throw 'Copy portable_env beside this task package, or supply -Environment.'
}
$runner = Join-Path $Environment 'run.ps1'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Environment launcher missing: $runner" }

$scriptArgs = @('--root', $packageRoot, '--interval',
                $IntervalSeconds.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture))
if ($RunDir) { $scriptArgs += @('--run-dir', $RunDir) }
if ($Follow) { $scriptArgs += '--follow' }
& $runner -Python $Python -Script $monitor -Offline -ScriptArguments $scriptArgs
exit $LASTEXITCODE
