<#
.SYNOPSIS
    Start the Cell Vision project review service and inference worker.

.DESCRIPTION
    Loads .env.production, verifies the read-only model bundle, starts one
    project service with its queue worker, and waits for /api/ready. The
    service is hidden by default on Windows; use -Foreground for diagnostics.
#>

[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [string]$Manifest = "",
    [switch]$Foreground,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

function Get-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$BasePath
    )
    $expanded = [Environment]::ExpandEnvironmentVariables($Value.Trim())
    if ([IO.Path]::IsPathRooted($expanded)) {
        return [IO.Path]::GetFullPath($expanded)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $expanded))
}

function Import-ProductionEnv {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Production environment file not found: $Path. Run setup_production.ps1 first."
    }
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -notmatch "^([^=]+)=(.*)$") { continue }
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

function Test-ModelBundle {
    param([Parameter(Mandatory = $true)][string]$Root)
    $required = @(
        (Join-Path $Root "models\teaching_classifier.pt"),
        (Join-Path $Root "models\multiplicity_classifier.pt"),
        (Join-Path $Root "v2\models\latest_instance_segmenter.pt")
    )
    $missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($missing.Count -gt 0) {
        $message = "Missing production model files:`n" + (($missing | ForEach-Object { "  $_" }) -join "`n")
        throw $message
    }
}

function ConvertTo-StartProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + $Value.Replace('"', '\"') + '"'
}

$scriptRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = $scriptRoot
} else {
    $InstallRoot = Get-AbsolutePath -Value $InstallRoot -BasePath (Get-Location).Path
}
$envPath = Join-Path $InstallRoot ".env.production"
Import-ProductionEnv -Path $envPath

$env:CELLVISION_PRODUCTION = "1"
$python = Join-Path $InstallRoot ".venv-production\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Production Python was not found: $python. Run setup_production.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = $env:CELLVISION_MANIFEST
}
if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = "artifacts/projects/active/project.json"
}
$manifestPath = Get-AbsolutePath -Value $Manifest -BasePath $InstallRoot
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Project manifest not found: $manifestPath"
}

$modelRoot = $env:CELLVISION_SOURCE_ARTIFACTS
if ([string]::IsNullOrWhiteSpace($modelRoot)) { $modelRoot = $env:CELLVISION_MODEL_ROOT }
if ([string]::IsNullOrWhiteSpace($modelRoot)) { throw "CELLVISION_SOURCE_ARTIFACTS or CELLVISION_MODEL_ROOT is required." }
$modelRoot = Get-AbsolutePath -Value $modelRoot -BasePath $InstallRoot
Test-ModelBundle -Root $modelRoot

$port = 8777
if (-not [string]::IsNullOrWhiteSpace($env:CELLVISION_PORT)) { $port = [int]$env:CELLVISION_PORT }
$bindHost = $env:CELLVISION_HOST
if ([string]::IsNullOrWhiteSpace($bindHost)) { $bindHost = "127.0.0.1" }
$workerDevice = $env:CELLVISION_WORKER_DEVICE
if ([string]::IsNullOrWhiteSpace($workerDevice)) { $workerDevice = "auto" }

$runtimeOutput = & $python -m cellvision runtime-info --device $workerDevice 2>&1
if ($LASTEXITCODE -ne 0) { throw "Runtime probe failed:`n$($runtimeOutput -join "`n")" }
$runtime = (($runtimeOutput -join "`n") | ConvertFrom-Json)
Write-Host "Cell Vision runtime: $($runtime.label)" -ForegroundColor Cyan
if ($runtime.fallback_reason) { Write-Host "Runtime note: $($runtime.fallback_reason)" -ForegroundColor DarkYellow }

$listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $commandLine = if ($null -ne $owner) { [string]$owner.CommandLine } else { "" }
    $isCellVision = $commandLine -match "cellvision" -and $commandLine -match "review-project" -and $commandLine -match ("--port\s+" + [regex]::Escape([string]$port))
    if ($Restart -and $isCellVision) {
        Write-Host "Stopping existing Cell Vision service (PID $($listener.OwningProcess)) ..." -ForegroundColor DarkYellow
        Stop-Process -Id $listener.OwningProcess -Force
    } elseif (-not $isCellVision) {
        throw "Port $port is already used by another process. Choose another port or stop that service explicitly."
    } else {
        throw "Cell Vision is already listening on port $port. Use -Restart to replace only that matching process."
    }
}
if ($Restart -and $listeners.Count -gt 0) { Start-Sleep -Milliseconds 700 }

$arguments = @(
    "-m", "cellvision", "review-project",
    "--manifest", $manifestPath,
    "--host", $bindHost,
    "--port", [string]$port,
    "--worker-device", $workerDevice
)
$allowRemote = $env:CELLVISION_ALLOW_REMOTE -eq "1"
if ($allowRemote) { $arguments += "--allow-remote" }

if ($Foreground) {
    Write-Host "Starting Cell Vision in foreground ..." -ForegroundColor Green
    & $python @arguments
    exit $LASTEXITCODE
}

$logRoot = $env:CELLVISION_LOG_ROOT
if ([string]::IsNullOrWhiteSpace($logRoot)) { $logRoot = Join-Path $InstallRoot "logs" }
$logRoot = Get-AbsolutePath -Value $logRoot -BasePath $InstallRoot
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stdout = Join-Path $logRoot "cellvision-production.stdout.log"
$stderr = Join-Path $logRoot "cellvision-production.stderr.log"
$quotedArguments = @($arguments | ForEach-Object { ConvertTo-StartProcessArgument -Value ([string]$_) })
$process = Start-Process -FilePath $python -ArgumentList $quotedArguments -WorkingDirectory $InstallRoot `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

$pidPath = Join-Path $logRoot "cellvision-production.pid"
[IO.File]::WriteAllText($pidPath, [string]$process.Id, (New-Object Text.UTF8Encoding($false)))
Write-Host "Cell Vision service started (PID $($process.Id)); waiting for readiness ..." -ForegroundColor Green

$readyUrl = "http://127.0.0.1:$port/api/ready"
$ready = $false
for ($attempt = 1; $attempt -le 60; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri $readyUrl -Method Get -TimeoutSec 3
        if ($response.status -eq "ready") {
            $ready = $true
            Write-Host "Ready: $readyUrl" -ForegroundColor Green
            break
        }
    } catch {
        if ($process.HasExited) { break }
    }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    $stdoutTail = if (Test-Path -LiteralPath $stdout) { Get-Content -LiteralPath $stdout -Tail 30 } else { @() }
    $stderrTail = if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Tail 30 } else { @() }
    Write-Host "Service did not become ready. stdout: $stdout" -ForegroundColor Red
    Write-Host "stderr: $stderr" -ForegroundColor Red
    if ($stdoutTail.Count -gt 0) { Write-Host ($stdoutTail -join "`n") }
    if ($stderrTail.Count -gt 0) { Write-Host ($stderrTail -join "`n") }
    exit 1
}
