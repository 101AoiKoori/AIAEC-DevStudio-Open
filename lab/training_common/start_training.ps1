<#
.SYNOPSIS
  Validate this package, start TensorBoard, and run its training command.
.DESCRIPTION
  PowerShell must be started on Windows with execution policy permitting this
  script. The optional Python argument selects an existing interpreter; without
  it, the package virtual environment is preferred, followed by python on PATH.
  TensorBoard is a hidden child process and is stopped when this command exits.
  After successful training, press Enter to stop it. Use -NoWait in automation.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 6006,
    [string]$Python = '',
    [switch]$NoWait
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$trainScript = Join-Path $packageRoot 'train.py'
$tensorboardScript = Join-Path $packageRoot 'tensorboard_log.py'
$tensorboardProcess = $null
$locationPushed = $false
$exitCode = 1

try {
    foreach ($path in @($trainScript, $tensorboardScript)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required package file is missing: $path"
        }
    }

    if (-not $Python) {
        $packagePython = Join-Path $packageRoot '.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $packagePython -PathType Leaf) {
            $Python = $packagePython
        } else {
            $Python = 'python'
        }
    }
    $pythonCommand = Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $pythonCommand) {
        throw 'Python was not found. Install the dependencies in requirements.txt or use -Python to select an interpreter.'
    }
    $pythonExe = $pythonCommand.Source
    Push-Location -LiteralPath $packageRoot
    $locationPushed = $true
    Write-Host "Python: $pythonExe"
    Write-Host 'Checking Python dependencies...'
    & $pythonExe -c 'import torch, ultralytics, tensorboard, yaml, matplotlib'
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency check failed. Install requirements.txt using $pythonExe."
    }

    Write-Host 'Checking the dataset and model...'
    & $pythonExe $trainScript check
    if ($LASTEXITCODE -ne 0) {
        $exitCode = $LASTEXITCODE
        throw "Dataset or model check failed with exit code $exitCode."
    }

    # Reserve and release the requested loopback port before launching our
    # child; check the child again below in case another process wins the race.
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try {
        $listener.Start()
    } catch {
        throw "Port $Port is unavailable. Choose a free port with -Port, for example -Port 6007."
    } finally {
        $listener.Stop()
    }

    $logDir = Join-Path $packageRoot 'runs\_launcher'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $runTag = '{0}-{1}' -f $PID, [DateTime]::UtcNow.Ticks
    $stdoutPath = Join-Path $logDir "tensorboard-$runTag.out.log"
    $stderrPath = Join-Path $logDir "tensorboard-$runTag.err.log"
    # Start-Process joins ArgumentList elements; quote the script path so a
    # copied package inside a directory containing spaces remains usable.
    $quotedScript = '"' + $tensorboardScript + '"'
    $tensorboardProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @('-u', $quotedScript, '--host', '127.0.0.1', '--port', "$Port") `
        -WorkingDirectory $packageRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    $ready = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        $tensorboardProcess.Refresh()
        if ($tensorboardProcess.HasExited) {
            throw "TensorBoard exited before it was ready. See $stderrPath"
        }
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $connection = $client.ConnectAsync('127.0.0.1', $Port)
            if ($connection.Wait(200) -and $client.Connected) {
                $ready = $true
                break
            }
        } catch {
            # Still starting: retry until the bounded deadline.
        } finally {
            $client.Dispose()
        }
        Start-Sleep -Milliseconds 200
    }
    if (-not $ready) {
        throw "TensorBoard did not become ready within 30 seconds. See $stderrPath"
    }

    Write-Host "TensorBoard: http://127.0.0.1:$Port/"
    Write-Host 'Starting training. Press Ctrl+C to stop.'
    & $pythonExe -u $trainScript train
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Host "Training exited with code $exitCode."
    } elseif (-not $NoWait) {
        Write-Host 'Training finished. Review TensorBoard and press Enter to stop it.'
        [void][Console]::ReadLine()
    }
} catch {
    [Console]::Error.WriteLine("ERROR: $($_.Exception.Message)")
} finally {
    if ($null -ne $tensorboardProcess) {
        $tensorboardProcess.Refresh()
        if (-not $tensorboardProcess.HasExited) {
            Stop-Process -Id $tensorboardProcess.Id -ErrorAction SilentlyContinue
            [void]$tensorboardProcess.WaitForExit(5000)
        }
        $tensorboardProcess.Dispose()
    }
    if ($locationPushed) {
        Pop-Location
    }
}
exit $exitCode
