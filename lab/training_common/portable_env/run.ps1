[CmdletBinding()]
param(
    [string]$Script,
    [string]$Module,
    [switch]$Check,
    [switch]$RequireCuda,
    [switch]$Offline,
    [string]$Report,
    [string]$Model,
    [string]$Python,
    [Alias('Arguments')]
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArguments
)
$ErrorActionPreference = 'Stop'
$modeCount = 0
foreach ($mode in @([bool]$Script, [bool]$Module, [bool]$Check)) { if ($mode) { $modeCount++ } }
if ($modeCount -ne 1) { throw 'Choose exactly one of -Check, -Script or -Module.' }
$manifestPath = Join-Path $PSScriptRoot 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw 'Copy a built environment bundle, including manifest.json and site-packages.' }
$bundleOptions = [System.Collections.Generic.List[object]]::new()
foreach ($bundleRoot in @($PSScriptRoot, (Join-Path $PSScriptRoot 'cp313'), (Join-Path $PSScriptRoot 'cp314'))) {
    $candidateManifest = Join-Path $bundleRoot 'manifest.json'
    if (Test-Path -LiteralPath $candidateManifest -PathType Leaf) {
        $bundleOptions.Add([pscustomobject]@{ Root = $bundleRoot; Manifest = (Get-Content -LiteralPath $candidateManifest -Raw | ConvertFrom-Json) })
    }
}
$candidates = [System.Collections.Generic.List[string]]::new()
if ($Python) { $candidates.Add($Python) }
elseif ($env:AIAEC_PYTHON) { $candidates.Add($env:AIAEC_PYTHON) }
else {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $listing = & $launcher.Source -0p 2>$null
        foreach ($line in $listing) {
            if ($line -match '([A-Za-z]:\\.*python(?:\d+(?:\.\d+)?)?\.exe)\s*$') { $candidates.Add($Matches[1]) }
        }
    }
    foreach ($command in @(Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
        if ($command.Source -and $command.Source -notmatch '\\WindowsApps\\') { $candidates.Add($command.Source) }
    }
    foreach ($registryRoot in @('HKCU:\Software\Python', 'HKLM:\Software\Python', 'HKLM:\Software\WOW6432Node\Python')) {
        if (Test-Path -LiteralPath $registryRoot) {
            foreach ($key in @(Get-ChildItem -LiteralPath $registryRoot -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -eq 'InstallPath' })) {
                $location = $key.GetValue('ExecutablePath')
                if (-not $location) {
                    $baseLocation = $key.GetValue('')
                    if ($baseLocation) { $location = Join-Path $baseLocation 'python.exe' }
                }
                if ($location) { $candidates.Add($location) }
            }
        }
    }
}
$probe = "import json,sys,sysconfig,struct; print(json.dumps({'implementation':sys.implementation.name,'version':list(sys.version_info[:2]),'bits':struct.calcsize('P')*8,'platform':sysconfig.get_platform(),'debug':bool(sysconfig.get_config_var('Py_DEBUG')),'free_threaded':bool(sysconfig.get_config_var('Py_GIL_DISABLED'))}))"
$selected = $null
$selectedBundle = $null
$failures = [System.Collections.Generic.List[string]]::new()
foreach ($candidate in @($candidates | Select-Object -Unique)) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { $failures.Add("$candidate : missing"); continue }
    try {
        $raw = & $candidate -I -S -B -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'interpreter probe failed' }
        $info = $raw | ConvertFrom-Json
        foreach ($bundle in $bundleOptions) {
        $manifest = $bundle.Manifest
        if ($info.implementation -ne $manifest.python_implementation -or
            $info.version[0] -ne $manifest.python_version[0] -or $info.version[1] -ne $manifest.python_version[1] -or
            $info.bits -ne $manifest.bits -or $info.platform -ne $manifest.platform -or $info.debug -or $info.free_threaded) {
            continue
        }
        $selected = $candidate
        $selectedBundle = $bundle.Root
        break
        }
        if ($selected) { break }
        throw "incompatible with available bundles: $raw"
    } catch { $failures.Add("$candidate : $($_.Exception.Message)") }
}
if (-not $selected) {
    throw ("Compatible Windows x64 CPython for an available dependency bundle was not found. Install the matching interpreter or pass -Python; no dependency installation is attempted. Candidates: " + ($failures -join '; '))
}
$invokeArgs = @('-I', '-S', '-B', (Join-Path $selectedBundle 'bootstrap.py'))
if ($Check) { $invokeArgs += '--check' }
elseif ($Script) { $invokeArgs += @('--script', (Resolve-Path -LiteralPath $Script).Path) }
elseif ($Module) { $invokeArgs += @('--module', $Module) }
else { throw 'Choose -Check, -Script or -Module.' }
if ($RequireCuda) { $invokeArgs += '--require-cuda' }
if ($Offline) { $invokeArgs += '--offline' }
if ($Report) { $invokeArgs += @('--report', $Report) }
if ($Model) { $invokeArgs += @('--model', (Resolve-Path -LiteralPath $Model).Path) }
if ($ScriptArguments) { $invokeArgs += '--'; $invokeArgs += $ScriptArguments }
& $selected @invokeArgs
exit $LASTEXITCODE
