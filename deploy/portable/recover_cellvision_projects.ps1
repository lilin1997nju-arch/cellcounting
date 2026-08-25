<# Safely restore project manifests and force-refresh the Cell Vision homepage catalog. #>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [ValidateRange(15, 600)][int]$StartupTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$projectsRoot = Join-Path $workspaceRoot "Projects"
$logsRoot = Join-Path $workspaceRoot "Logs"
$python = Join-Path $applicationRoot "Python312\python.exe"
$environmentPath = Join-Path $applicationRoot ".env.production"
$recoveryScript = Join-Path $deploymentRoot "repair_project_manifests.py"
$dailyLauncher = Join-Path $deploymentRoot "start_cellvision_platform.ps1"
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null
$logPath = Join-Path $logsRoot "cellvision-project-recovery.log"

function Write-RecoveryStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ("[{0}] {1}" -f $timestamp, $Message) -ForegroundColor $Color
    try {
        Add-Content -LiteralPath $logPath -Value ("[{0}] {1}" -f $timestamp, $Message) -Encoding UTF8
    } catch {}
}

trap {
    Write-RecoveryStatus ("ERROR: " + $_.Exception.Message) Red
    Write-RecoveryStatus "No project data was deleted. Review the log above before retrying." Yellow
    exit 1
}

function Get-ConfiguredPort {
    if ($Port -gt 0) { return $Port }
    if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
        $line = Get-Content -LiteralPath $environmentPath | Where-Object {
            $_ -match '^CELLVISION_PORT=(\d+)\s*$'
        } | Select-Object -First 1
        if ($line -match '^CELLVISION_PORT=(\d+)\s*$') { return [int]$Matches[1] }
    }
    return 8777
}

Write-RecoveryStatus "Starting Cell Vision project-list recovery." Cyan
foreach ($required in @($python, $environmentPath, $recoveryScript, $dailyLauncher)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "This is not a complete Cell Vision installation. Missing: $required"
    }
}
New-Item -ItemType Directory -Force -Path $projectsRoot | Out-Null

Write-RecoveryStatus "Backing up the current project metadata before recovery ..." Cyan
$recoveryJson = & $python $recoveryScript --projects-root $projectsRoot --backup --recover
if ($LASTEXITCODE -ne 0) {
    throw "The recovery utility exited with code $LASTEXITCODE."
}
try {
    $result = $recoveryJson | ConvertFrom-Json
} catch {
    throw "The recovery utility returned an unreadable result: $recoveryJson"
}

$backupPath = [string]$result.backup.backup_path
Write-RecoveryStatus "Metadata backup: $backupPath" DarkGray
$restored = @($result.recovery.restored_from_backup)
$reconstructed = @($result.recovery.reconstructed_from_catalog)
$skipped = @($result.recovery.skipped)

foreach ($project in $restored) {
    Write-RecoveryStatus "Restored exact project metadata from backup: $project" Green
}
foreach ($project in $reconstructed) {
    Write-RecoveryStatus "Reconstructed project metadata from complete catalogued artifacts: $project" Green
}
foreach ($entry in $skipped) {
    Write-RecoveryStatus ("Skipped {0}: {1}" -f $entry.project, $entry.reason) Yellow
}
if ($restored.Count -eq 0 -and $reconstructed.Count -eq 0) {
    Write-RecoveryStatus "No missing recoverable project manifest was found; existing manifests will still be re-indexed." DarkGray
}

Write-RecoveryStatus "Checking the service and compute worker before refreshing the homepage ..." Cyan
& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $dailyLauncher `
    -Port $Port -StartupTimeoutSeconds $StartupTimeoutSeconds -NoOpen
if ($LASTEXITCODE -ne 0) {
    throw "The Cell Vision service could not be made ready."
}

$readyPort = Get-ConfiguredPort
$baseUrl = "http://127.0.0.1:$readyPort/"
$catalog = Invoke-RestMethod -Uri ($baseUrl + "api/catalog/status") -TimeoutSec 30
$projectCount = 0
$plateCount = 0
try { $projectCount = [int]$catalog.projects } catch {}
try { $plateCount = [int]$catalog.plates } catch {}
Write-RecoveryStatus "Homepage catalog refreshed: $projectCount project(s), $plateCount plate(s)." Green
if ($skipped.Count -gt 0) {
    Write-RecoveryStatus (
        "$($skipped.Count) folder(s) could not be safely recovered. Their data was left unchanged; " +
        "see the warnings and log: $logPath"
    ) Yellow
} else {
    Write-RecoveryStatus "Project-list recovery completed without skipped folders." Green
}
Write-RecoveryStatus "Refresh the Cell Vision homepage in the browser. Log: $logPath" Cyan
exit 0
