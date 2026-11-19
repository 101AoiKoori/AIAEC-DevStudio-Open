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
$bundleOptions = [System.Collections.Generic.List[object]]::new()
$bundleFailures = [System.Collections.Generic.List[string]]::new()
foreach ($bundleRoot in @($PSScriptRoot, (Join-Path $PSScriptRoot 'cp314'))) {
    # An absent optional bundle is normal; an incomplete present one must not
    # prevent another compatible bundle from being selected.
    if ($bundleRoot -ne $PSScriptRoot -and -not (Test-Path -LiteralPath $bundleRoot -PathType Container)) { continue }
    try {
        $candidateManifest = Join-Path $bundleRoot 'manifest.json'
        if (-not (Test-Path -LiteralPath $candidateManifest -PathType Leaf)) { throw 'manifest.json missing' }
        if (-not (Test-Path -LiteralPath (Join-Path $bundleRoot 'bootstrap.py') -PathType Leaf)) { throw 'bootstrap.py missing' }
        if (-not (Test-Path -LiteralPath (Join-Path $bundleRoot 'site-packages') -PathType Container)) { throw 'site-packages directory missing' }
        $manifest = Get-Content -LiteralPath $candidateManifest -Raw | ConvertFrom-Json
        if ($null -eq $manifest -or -not $manifest.python_implementation -or
            @($manifest.python_version).Count -lt 2 -or -not $manifest.bits -or -not $manifest.platform) {
            throw 'manifest lacks Python compatibility fields'
        }
        $bundleOptions.Add([pscustomobject]@{ Root = $bundleRoot; Manifest = $manifest })
    } catch {
        $failure = "$bundleRoot : $($_.Exception.Message)"
        $bundleFailures.Add($failure)
        Write-Warning "Skipping unavailable portable bundle: $failure"
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
        if ($info.version[0] -ne 3 -or $info.version[1] -notin @(12, 14)) {
            throw "Unsupported Python $($info.version -join '.'); supported versions are CPython 3.12 and 3.14."
        }
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
    throw ("Compatible Windows x64 CPython 3.12 or 3.14 for an available dependency bundle was not found. Python 3.13 is unsupported. Install the matching interpreter or pass -Python; no dependency installation is attempted. Bundles: " + ($bundleFailures -join '; ') + ". Candidates: " + ($failures -join '; '))
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
