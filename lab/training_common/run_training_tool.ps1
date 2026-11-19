param([string]$Entry, [string[]]$ScriptArguments, [string]$Workspace, [switch]$Portable)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $Entry)
if ($Workspace) {
    $venv = Join-Path $Workspace '.venv\Scripts\python.exe'
    if ((Test-Path -LiteralPath $venv -PathType Leaf) -and -not $env:AIAEC_PYTHON -and -not $Portable) {
        & $venv -X utf8 -B $Entry @ScriptArguments
        exit $LASTEXITCODE
    }
    $runner = Join-Path $Workspace 'training_packages\portable_env\run.ps1'
} else {
    $runner = Join-Path $root 'portable_env\run.ps1'
    if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
        $runner = Join-Path (Split-Path -Parent $root) 'portable_env\run.ps1'
    }
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw 'Python environment missing. Use the workspace .venv or place portable_env beside the task package.'
}
& $runner -Script $Entry -Offline -ScriptArguments $ScriptArguments
exit $LASTEXITCODE
