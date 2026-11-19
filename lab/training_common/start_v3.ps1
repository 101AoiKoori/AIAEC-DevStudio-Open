[CmdletBinding()]
param(
    [string]$Environment = '',
    [string]$Python = '',
    [ValidateRange(1,65535)][int]$Port = 6006,
    [string]$Config = 'config/train.yaml',
    [string]$Resume = '',
    [double]$WallSeconds = 0,
    [ValidateRange(1,2147483647)][int]$Epochs,
    [switch]$NoWait,
    [switch]$NoTensorBoard
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$launch = Join-Path $PSScriptRoot 'launch.py'
$scriptArgs = @('--root', $packageRoot, '--port', "$Port")
if ($Resume) { $scriptArgs += @('--resume', $Resume) }
if (-not $Resume -or $PSBoundParameters.ContainsKey('Config')) {
    $scriptArgs += @('--config', $Config)
}
if ($NoWait) { $scriptArgs += '--no-wait' }
if ($NoTensorBoard) { $scriptArgs += '--no-tensorboard' }
if ($PSBoundParameters.ContainsKey('Epochs')) { $scriptArgs += @('--epochs', "$Epochs") }
if ($PSBoundParameters.ContainsKey('WallSeconds')) {
    $scriptArgs += @('--wall-seconds', $WallSeconds.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture))
}
if (-not $Environment) {
    foreach ($candidate in @((Join-Path $packageRoot 'portable_env'), (Join-Path (Split-Path -Parent $packageRoot) 'portable_env'))) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'run.ps1') -PathType Leaf) {
            $Environment = $candidate
            break
        }
    }
}
if (-not $Environment) {
    throw 'Copy the portable_env bundle beside this task package, or supply -Environment. No target installation is performed.'
}
$runner = Join-Path $Environment 'run.ps1'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Environment launcher missing: $runner" }
& $runner -Python $Python -Script $launch -Offline -ScriptArguments $scriptArgs
exit $LASTEXITCODE
