param(
    [string]$Manifest = "artifacts/projects/ql2603/project.json",
    [int]$Port = 8777
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$stdout = Join-Path $root "artifacts\logs\review-project-$Port.stdout.log"
$stderr = Join-Path $root "artifacts\logs\review-project-$Port.stderr.log"

# Only stop a process that is both listening on the requested port and running
# this project's review-project command. Never kill an unrelated service.
$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($listener in @($listeners)) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($null -ne $process -and $process.CommandLine -match "cellvision review-project" -and $process.CommandLine -match "--port\s+$Port") {
        Stop-Process -Id $listener.OwningProcess -Force
    }
}

Start-Sleep -Milliseconds 500
Start-Process -FilePath $python `
    -ArgumentList @("-m", "cellvision", "review-project", "--manifest", $Manifest, "--host", "127.0.0.1", "--port", "$Port") `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden | Out-Null

Write-Host "Cell Vision project server restarted on http://127.0.0.1:$Port/"

