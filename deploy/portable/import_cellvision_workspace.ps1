<# Import a historical Workspace while the machine-wide service is stopped. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceWorkspace,
    [int]$StartupTimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionDesktopProduction"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$targetWorkspace = Join-Path $deploymentRoot "Workspace"
$projectsRoot = Join-Path $targetWorkspace "Projects"
$logsRoot = Join-Path $targetWorkspace "Logs"
$python = Join-Path $applicationRoot "Python312\python.exe"
$importScript = Join-Path $deploymentRoot "import_cellvision_workspace.py"
$recoveryScript = Join-Path $deploymentRoot "repair_project_manifests.py"
$environmentPath = Join-Path $applicationRoot ".env.production"
New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null
$logPath = Join-Path $logsRoot "cellvision-workspace-import.log"

function Write-ImportStatus {
    param([string]$Message, [ConsoleColor]$Color = [ConsoleColor]::Gray)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ("[{0}] {1}" -f $timestamp, $Message) -ForegroundColor $Color
    try { Add-Content -LiteralPath $logPath -Value ("[{0}] {1}" -f $timestamp, $Message) -Encoding UTF8 } catch {}
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ConfiguredPort {
    if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
        $line = Get-Content -LiteralPath $environmentPath | Where-Object { $_ -match '^CELLVISION_PORT=(\d+)\s*$' } | Select-Object -First 1
        if ($line -match '^CELLVISION_PORT=(\d+)\s*$') { return [int]$Matches[1] }
    }
    return 8777
}

if (-not (Test-Administrator)) { throw "Administrator permission is required to import a Workspace safely." }
foreach ($required in @($python, $importScript, $recoveryScript, $SourceWorkspace)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required import path is missing: $required" }
}
$source = [IO.Path]::GetFullPath($SourceWorkspace)
if (-not (Test-Path -LiteralPath (Join-Path $source "Projects") -PathType Container)) {
    throw "The selected folder is not a Cell Vision Workspace: $source"
}
if ([string]::Equals($source.TrimEnd('\'), $targetWorkspace.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw "The selected Workspace is already the active Workspace. Use project repair instead."
}

$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
$restartService = $null -ne $service -and $service.Status -ne "Stopped"
try {
    if ($restartService) {
        Write-ImportStatus "Stopping $serviceName for a consistent Workspace import ..." Cyan
        Stop-Service -Name $serviceName -Force
        (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(60))
    }
    Write-ImportStatus "Importing historical Workspace from: $source" Cyan
    $json = & $python $importScript --source-workspace $source --target-workspace $targetWorkspace
    if ($LASTEXITCODE -ne 0) { throw "Workspace import process exited with code $LASTEXITCODE." }
    $result = $json | ConvertFrom-Json
    Write-ImportStatus ("Imported projects: " + @($result.imported_projects).Count) Green
    foreach ($item in @($result.skipped_projects)) {
        Write-ImportStatus ("Skipped {0}: {1}" -f $item.project, $item.reason) Yellow
    }
    Write-ImportStatus ("Rebased JSON metadata files: " + [int]$result.rebased_json_files) DarkGray
} finally {
    if ($null -ne $service) {
        $current = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        if ($null -ne $current -and $current.Status -eq "Stopped") {
            Write-ImportStatus "Starting $serviceName ..." Cyan
            Start-Service -Name $serviceName
        }
    }
}

if ($null -ne $service) {
    $port = Get-ConfiguredPort
    $ready = $false
    for ($attempt = 0; $attempt -lt $StartupTimeoutSeconds; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/ready" -TimeoutSec 2
            if ($response.status -eq "ready") { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "$serviceName did not become ready after the Workspace import." }
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/catalog/status" -TimeoutSec 30
}
Write-ImportStatus "Workspace import and project re-index completed. Log: $logPath" Green
exit 0
