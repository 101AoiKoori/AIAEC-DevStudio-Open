[CmdletBinding()]
param(
    [string]$Environment = '',
    [string]$Python = '',
    [ValidateRange(1,65535)][int]$Port = 6006,
    [string]$Config = 'config/train.yaml',
    [switch]$NoWait
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$launch = Join-Path $PSScriptRoot 'launch.py'
$scriptArgs = @('--root', $packageRoot, '--port', "$Port", '--config', $Config)
if ($NoWait) { $scriptArgs += '--no-wait' }
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
