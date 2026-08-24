<# Restore the backup made by the most recently installed Cell Vision update. #>
[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [int]$Port = 0,
    [switch]$SkipServiceRestart
)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionProduction"
$packageRoot = [IO.Path]::GetFullPath($PSScriptRoot)

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ServiceApplicationRoot {
    $service = Get-CimInstance Win32_Service -Filter "Name='$serviceName'" -ErrorAction SilentlyContinue
    if ($null -eq $service) { return $null }
    $pathName = [string]$service.PathName
    $executable = ""
    if ($pathName -match '^\s*"([^"]+)"') { $executable = $Matches[1] }
    elseif ($pathName -match '^\s*([^\s]+)') { $executable = $Matches[1] }
    if ([string]::IsNullOrWhiteSpace($executable)) { return $null }
    $runtimeRoot = Split-Path -Parent ([IO.Path]::GetFullPath($executable))
    if ((Split-Path -Leaf $runtimeRoot) -ne "Python312") { return $null }
    return [IO.Path]::GetFullPath((Split-Path -Parent $runtimeRoot))
}

function Resolve-ApplicationRoot {
    param([string]$RequestedRoot)
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $candidate = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($RequestedRoot))
        if (Test-Path -LiteralPath (Join-Path $candidate "Application\Python312\python.exe") -PathType Leaf) {
            return [IO.Path]::GetFullPath((Join-Path $candidate "Application"))
        }
        return $candidate
    }
    $serviceRoot = Get-ServiceApplicationRoot
    if ($null -ne $serviceRoot) { return $serviceRoot }
    $parent = [IO.Path]::GetFullPath((Split-Path -Parent $packageRoot))
    foreach ($candidate in @((Join-Path $parent "Application"), $parent)) {
        if (Test-Path -LiteralPath (Join-Path $candidate "Python312\python.exe") -PathType Leaf) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }
    throw "Unable to locate the Cell Vision application root. Place this update folder in the installation root or pass -InstallRoot."
}

function Resolve-ChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )
    if ([IO.Path]::IsPathRooted($RelativePath)) { throw "Backup path must be relative: $RelativePath" }
    $normalizedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $resolved = [IO.Path]::GetFullPath((Join-Path $Root $RelativePath))
    if (-not $resolved.StartsWith($normalizedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Backup path escapes its root: $RelativePath"
    }
    return $resolved
}

function Get-ConfiguredPort {
    param([string]$Root, [int]$RequestedPort)
    if ($RequestedPort -gt 0) { return $RequestedPort }
    $environmentPath = Join-Path $Root ".env.production"
    if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
        $line = Get-Content -LiteralPath $environmentPath | Where-Object { $_ -match '^CELLVISION_PORT=(\d+)' } | Select-Object -First 1
        if ($line -match '^CELLVISION_PORT=(\d+)') { return [int]$Matches[1] }
    }
    return 8777
}

function Wait-ServiceReady {
    param([int]$ReadyPort)
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$ReadyPort/api/ready" -TimeoutSec 2
            if ($response.status -eq "ready") { return }
        } catch {}
        Start-Sleep -Seconds 1
    }
    throw "Cell Vision service did not become ready on port $ReadyPort."
}

if (-not $SkipServiceRestart -and -not (Test-Administrator)) {
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    if (-not [string]::IsNullOrWhiteSpace($InstallRoot)) {
        $arguments += " -InstallRoot `"$InstallRoot`""
    }
    if ($Port -gt 0) { $arguments += " -Port $Port" }
    $elevated = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments -WorkingDirectory $packageRoot -Wait -PassThru
    exit $elevated.ExitCode
}

$applicationRoot = Resolve-ApplicationRoot -RequestedRoot $InstallRoot
$installedRecordPath = Join-Path $applicationRoot "CELLVISION_UPDATE.json"
if (-not (Test-Path -LiteralPath $installedRecordPath -PathType Leaf)) {
    throw "No installed update record was found: $installedRecordPath"
}
$installedRecord = Get-Content -LiteralPath $installedRecordPath -Raw -Encoding UTF8 | ConvertFrom-Json
$backupRoot = [IO.Path]::GetFullPath([string]$installedRecord.backup_root)
$normalizedApplicationRoot = [IO.Path]::GetFullPath($applicationRoot).TrimEnd('\') + '\'
if (-not $backupRoot.StartsWith($normalizedApplicationRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The recorded backup is outside the application root: $backupRoot"
}
$backupManifestPath = Join-Path $backupRoot "BACKUP.json"
if (-not (Test-Path -LiteralPath $backupManifestPath -PathType Leaf)) {
    throw "Update backup manifest is missing: $backupManifestPath"
}
$backupManifest = Get-Content -LiteralPath $backupManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$backupManifest.format -ne "cellvision-update-backup" -or [int]$backupManifest.version -ne 1) {
    throw "Unsupported Cell Vision backup manifest."
}

$restoreFiles = New-Object System.Collections.Generic.List[object]
foreach ($file in $backupManifest.files) {
    $relative = [string]$file.path
    $backupPath = Resolve-ChildPath -Root $backupRoot -RelativePath $relative
    $targetPath = Resolve-ChildPath -Root $applicationRoot -RelativePath $relative
    if (-not (Test-Path -LiteralPath $backupPath -PathType Leaf)) { throw "Backup file is missing: $relative" }
    $restoreFiles.Add([pscustomobject]@{ backup_path = $backupPath; target_path = $targetPath })
}

$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
if (-not $SkipServiceRestart -and $null -eq $service) { throw "The $serviceName service is not installed." }
$readyPort = Get-ConfiguredPort -Root $applicationRoot -RequestedPort $Port

if (-not $SkipServiceRestart -and $service.Status -ne "Stopped") {
    Write-Host "Stopping $serviceName ..." -ForegroundColor DarkYellow
    Stop-Service -Name $serviceName -Force
    (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
}
foreach ($file in $restoreFiles) {
    Copy-Item -LiteralPath $file.backup_path -Destination $file.target_path -Force
}
if (-not $SkipServiceRestart) {
    Write-Host "Starting $serviceName ..." -ForegroundColor Cyan
    Start-Service -Name $serviceName
    (Get-Service -Name $serviceName).WaitForStatus("Running", [TimeSpan]::FromSeconds(30))
    Wait-ServiceReady -ReadyPort $readyPort
}

$rollbackRecord = [pscustomobject]@{
    format = "cellvision-rolled-back-update"
    version = 1
    package_id = [string]$installedRecord.package_id
    rolled_back_at = (Get-Date).ToUniversalTime().ToString("o")
    backup_root = $backupRoot
    file_count = $restoreFiles.Count
}
$rollbackRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $installedRecordPath -Encoding UTF8
($rollbackRecord | ConvertTo-Json -Compress) | Add-Content -LiteralPath (Join-Path $applicationRoot "UPDATE_HISTORY.jsonl") -Encoding UTF8
Write-Host "Cell Vision update $($installedRecord.target_version) was rolled back successfully." -ForegroundColor Green
Write-Host "Application: $applicationRoot"
