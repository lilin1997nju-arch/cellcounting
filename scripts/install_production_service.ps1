<# Install and manage the machine-wide Cell Vision production service. #>
[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [int]$Port = 8777,
    [ValidateSet("install", "uninstall", "start", "stop", "status")]
    [string]$Action = "install"
)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionProduction"
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
} else {
    $InstallRoot = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($InstallRoot))
}
$basePython = Join-Path $InstallRoot "Python312\python.exe"
$manager = Join-Path $InstallRoot "scripts\manage_production_service.py"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Get-CellVisionService {
    return Get-Service -Name $serviceName -ErrorAction SilentlyContinue
}
function Invoke-ServiceManager {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) { throw "Service Python is missing: $basePython" }
    if (-not (Test-Path -LiteralPath $manager -PathType Leaf)) { throw "Service manager is missing: $manager" }
    & $basePython $manager @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Service manager failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')" }
}

if ($Action -eq "status") {
    $service = Get-CellVisionService
    if ($null -eq $service) { Write-Host "Cell Vision production service is not installed."; exit 1 }
    Write-Host "Cell Vision production service: $($service.Status)"
    exit 0
}
if (-not (Test-Administrator)) {
    throw "Administrator permission is required to manage the Cell Vision production service."
}
if ($Action -eq "stop") {
    $service = Get-CellVisionService
    if ($null -ne $service -and $service.Status -ne "Stopped") {
        Stop-Service -Name $serviceName -Force
        (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
    }
    exit 0
}
if ($Action -eq "uninstall") {
    $service = Get-CellVisionService
    if ($null -ne $service) {
        if ($service.Status -ne "Stopped") {
            Stop-Service -Name $serviceName -Force
            (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
        }
        Invoke-ServiceManager remove
    }
    exit 0
}
if ($Action -eq "start") {
    Start-Service -Name $serviceName
    (Get-Service -Name $serviceName).WaitForStatus("Running", [TimeSpan]::FromSeconds(30))
    exit 0
}

$existing = Get-CellVisionService
if ($null -ne $existing) {
    if ($existing.Status -ne "Stopped") {
        Stop-Service -Name $serviceName -Force
        (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
    }
    Invoke-ServiceManager remove
}
Invoke-ServiceManager --startup delayed install
& sc.exe failure $serviceName reset= 86400 actions= restart/5000/restart/15000/restart/60000 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Unable to configure service recovery actions." }
& sc.exe failureflag $serviceName 1 | Out-Null
Start-Service -Name $serviceName
(Get-Service -Name $serviceName).WaitForStatus("Running", [TimeSpan]::FromSeconds(30))

$ready = $false
for ($attempt = 0; $attempt -lt 90; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ready" -TimeoutSec 2
        if ($response.status -eq "ready") { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}
if (-not $ready) { throw "Cell Vision service started but did not become ready on port $Port." }
Write-Host "Cell Vision production service is installed, running, and ready." -ForegroundColor Green
