<# Back up and apply a Cell Vision incremental update, then restart the production service. #>
[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [int]$Port = 0,
    [switch]$SkipServiceRestart
)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionProduction"
$packageRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$manifestPath = Join-Path $packageRoot "UPDATE.json"

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
    if ([IO.Path]::IsPathRooted($RelativePath)) { throw "Update path must be relative: $RelativePath" }
    $normalizedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $resolved = [IO.Path]::GetFullPath((Join-Path $Root $RelativePath))
    if (-not $resolved.StartsWith($normalizedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Update path escapes its root: $RelativePath"
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

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Update manifest is missing: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([string]$manifest.format -ne "cellvision-incremental-update" -or [int]$manifest.version -ne 1) {
    throw "Unsupported Cell Vision update manifest."
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
if (-not (Test-Path -LiteralPath $applicationRoot -PathType Container)) {
    throw "Cell Vision application root does not exist: $applicationRoot"
}
$changes = New-Object System.Collections.Generic.List[object]
foreach ($file in $manifest.files) {
    $relative = [string]$file.path
    $payloadPath = Resolve-ChildPath -Root (Join-Path $packageRoot "payload") -RelativePath $relative
    $targetPath = Resolve-ChildPath -Root $applicationRoot -RelativePath $relative
    if (-not (Test-Path -LiteralPath $payloadPath -PathType Leaf)) { throw "Update payload is missing: $relative" }
    if (-not (Test-Path -LiteralPath $targetPath -PathType Leaf)) { throw "Production file is missing: $relative" }
    $changes.Add([pscustomobject]@{
        relative_path = $relative
        payload_path = $payloadPath
        target_path = $targetPath
    })
}

$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
if (-not $SkipServiceRestart -and $null -eq $service) {
    throw "The $serviceName service is not installed."
}
$serviceWasRunning = $null -ne $service -and $service.Status -ne "Stopped"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupRoot = Join-Path $applicationRoot ("update-backups\{0}-{1}" -f $timestamp, [string]$manifest.package_id)
$backupRecords = New-Object System.Collections.Generic.List[object]
$readyPort = Get-ConfiguredPort -Root $applicationRoot -RequestedPort $Port
try {
    if (-not $SkipServiceRestart -and $service.Status -ne "Stopped") {
        Write-Host "Stopping $serviceName ..." -ForegroundColor DarkYellow
        Stop-Service -Name $serviceName -Force
        (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
    }

    if ($changes.Count -gt 0) {
        New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
        foreach ($change in $changes) {
            $backupPath = Resolve-ChildPath -Root $backupRoot -RelativePath $change.relative_path
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backupPath) | Out-Null
            Copy-Item -LiteralPath $change.target_path -Destination $backupPath -Force
            $backupRecords.Add([pscustomobject]@{
                path = $change.relative_path
            })
        }
        [pscustomobject]@{
            format = "cellvision-update-backup"
            version = 1
            package_id = [string]$manifest.package_id
            application_root = $applicationRoot
            created_at = (Get-Date).ToUniversalTime().ToString("o")
            files = @($backupRecords | ForEach-Object { $_ })
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $backupRoot "BACKUP.json") -Encoding UTF8

        foreach ($change in $changes) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $change.target_path) | Out-Null
            Copy-Item -LiteralPath $change.payload_path -Destination $change.target_path -Force
        }
    }

    if (-not $SkipServiceRestart) {
        Write-Host "Starting $serviceName ..." -ForegroundColor Cyan
        Start-Service -Name $serviceName
        (Get-Service -Name $serviceName).WaitForStatus("Running", [TimeSpan]::FromSeconds(30))
        Wait-ServiceReady -ReadyPort $readyPort
    }

    if ($changes.Count -gt 0) {
        $updateRecord = [pscustomobject]@{
            format = "cellvision-installed-update"
            version = 1
            package_id = [string]$manifest.package_id
            target_version = [string]$manifest.target_version
            production_baseline = [string]$manifest.production_baseline
            installed_at = (Get-Date).ToUniversalTime().ToString("o")
            backup_root = $backupRoot
            file_count = $changes.Count
        }
        $updateRecord | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $applicationRoot "CELLVISION_UPDATE.json") -Encoding UTF8
        ($updateRecord | ConvertTo-Json -Compress) | Add-Content -LiteralPath (Join-Path $applicationRoot "UPDATE_HISTORY.jsonl") -Encoding UTF8
    }
    Write-Host "Cell Vision update $($manifest.target_version) is installed and the service is ready." -ForegroundColor Green
    Write-Host "Application: $applicationRoot"
    if ($changes.Count -gt 0) { Write-Host "Backup:      $backupRoot" }
} catch {
    $failure = $_
    if ($backupRecords.Count -gt 0) {
        Write-Host "Update failed; restoring the backup ..." -ForegroundColor Red
        if (-not $SkipServiceRestart) {
            $currentService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
            if ($null -ne $currentService -and $currentService.Status -ne "Stopped") {
                Stop-Service -Name $serviceName -Force
                (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
            }
        }
        foreach ($record in $backupRecords) {
            $backupPath = Resolve-ChildPath -Root $backupRoot -RelativePath $record.path
            $targetPath = Resolve-ChildPath -Root $applicationRoot -RelativePath $record.path
            if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
                Copy-Item -LiteralPath $backupPath -Destination $targetPath -Force
            }
        }
    }
    if (-not $SkipServiceRestart -and $serviceWasRunning) {
        $currentService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        if ($null -ne $currentService -and $currentService.Status -eq "Stopped") {
            Start-Service -Name $serviceName
            (Get-Service -Name $serviceName).WaitForStatus("Running", [TimeSpan]::FromSeconds(30))
            Wait-ServiceReady -ReadyPort $readyPort
        }
    }
    throw $failure
}
