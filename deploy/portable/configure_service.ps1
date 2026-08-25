<# Configure the copied portable production folder as a machine-wide service. #>
[CmdletBinding()]
param(
    [int]$Port = 8777,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [switch]$RepairService,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$python = Join-Path $applicationRoot "Python312\python.exe"
$serviceHost = Join-Path $applicationRoot "Python312\pythonservice.exe"
$recoveryScript = Join-Path $deploymentRoot "repair_project_manifests.py"
$serviceRuntimeDll = Get-ChildItem -LiteralPath (Join-Path $applicationRoot "Python312") `
    -Filter "pywintypes*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1

foreach ($required in @(
    $python,
    $serviceHost,
    (Join-Path $applicationRoot "src\cellvision"),
    (Join-Path $applicationRoot "ModelBundle"),
    (Join-Path $applicationRoot "scripts\setup_production.ps1"),
    $recoveryScript
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Portable deployment is incomplete: $required" }
}
if ($null -eq $serviceRuntimeDll) {
    throw "Portable Windows service runtime is incomplete: pywintypes DLL is missing beside python.exe."
}

Write-Host "Checking the portable Cell Vision runtime ..." -ForegroundColor Cyan
& $python -c "import cellvision, fastapi, torch, win32serviceutil; print(torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "The portable Python runtime is incomplete." }
if ($ValidateOnly) {
    Write-Output "PORTABLE_SERVICE_CONFIGURATION_VALIDATION_OK"
    exit 0
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Administrator permission is required to configure the machine-wide Cell Vision service."
}

$projectsRoot = Join-Path $workspaceRoot "Projects"
$databaseRoot = Join-Path $workspaceRoot "Database"
$logsRoot = Join-Path $workspaceRoot "Logs"
$inboxRoot = Join-Path $workspaceRoot "Inbox"
$manifest = Join-Path $projectsRoot "active\project.json"
foreach ($directory in @($workspaceRoot, $projectsRoot, $databaseRoot, $logsRoot, $inboxRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

Write-Host "Backing up project metadata before service configuration ..." -ForegroundColor Cyan
$backupOutput = & $python $recoveryScript --projects-root $projectsRoot --backup
if ($LASTEXITCODE -ne 0) { throw "Unable to back up project metadata before service configuration." }
Write-Host $backupOutput -ForegroundColor DarkGray

$setup = Join-Path $applicationRoot "scripts\setup_production.ps1"
$previousPipNoIndex = $env:PIP_NO_INDEX
$previousPipDisableVersionCheck = $env:PIP_DISABLE_PIP_VERSION_CHECK
try {
    $env:PIP_NO_INDEX = "1"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    & $setup -InstallRoot $applicationRoot -DataRoot $inboxRoot -ArtifactRoot $workspaceRoot `
        -ModelRoot (Join-Path $applicationRoot "ModelBundle") -DbRoot $databaseRoot `
        -LogRoot $logsRoot -Manifest $manifest -BasePython $python `
        -UseBasePythonRuntime -SkipDependencyInstall -Device $Device -Port $Port
    if ($LASTEXITCODE -ne 0) { throw "Cell Vision portable runtime configuration failed." }
} finally {
    $env:PIP_NO_INDEX = $previousPipNoIndex
    $env:PIP_DISABLE_PIP_VERSION_CHECK = $previousPipDisableVersionCheck
}

Write-Host "Using the existing administrator and LocalSystem permissions; no recursive ACL rewrite is needed." -ForegroundColor DarkGray
Write-Host "Ensuring the Cell Vision service is registered and ready ..." -ForegroundColor Cyan
$serviceInstaller = Join-Path $applicationRoot "scripts\install_production_service.ps1"
$existingService = Get-Service -Name "CellVisionProduction" -ErrorAction SilentlyContinue
$installService = $null -eq $existingService
if ($null -ne $existingService) {
    $serviceInfo = Get-CimInstance Win32_Service -Filter "Name='CellVisionProduction'" -ErrorAction Stop
    $pathName = [string]$serviceInfo.PathName
    $actualHost = ""
    if ($pathName -match '^\s*"([^"]+)"') { $actualHost = $Matches[1] }
    elseif ($pathName -match '^\s*([^\s]+)') { $actualHost = $Matches[1] }
    $sameHost = -not [string]::IsNullOrWhiteSpace($actualHost) -and [string]::Equals(
        [IO.Path]::GetFullPath($actualHost),
        [IO.Path]::GetFullPath($serviceHost),
        [StringComparison]::OrdinalIgnoreCase
    )
    if (-not $sameHost -and -not $RepairService) {
        throw (
            "CellVisionProduction belongs to a different installation: $actualHost. " +
            "Refusing to switch Workspace automatically. Use the daily launcher from that installation, " +
            "or rerun configure_service.ps1 with -RepairService after confirming the intended folder."
        )
    }
    $installService = -not $sameHost
}
if ($installService) {
    & $serviceInstaller -InstallRoot $applicationRoot -Port $Port -Action install
    if ($LASTEXITCODE -ne 0) { throw "Unable to install the Cell Vision production service." }
} else {
    if ($existingService.Status -ne "Stopped") {
        & $serviceInstaller -InstallRoot $applicationRoot -Port $Port -Action stop
        if ($LASTEXITCODE -ne 0) { throw "Unable to stop the existing Cell Vision production service." }
    }
    & $serviceInstaller -InstallRoot $applicationRoot -Port $Port -Action start
    if ($LASTEXITCODE -ne 0) { throw "Unable to start the existing Cell Vision production service." }
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ready" -TimeoutSec 2
            if ($response.status -eq "ready") { $ready = $true; break }
        } catch {}
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "Cell Vision service did not become ready on port $Port." }
}

$recoveryOutput = & $python $recoveryScript --projects-root $projectsRoot --recover
if ($LASTEXITCODE -ne 0) { throw "Project metadata recovery failed after service configuration." }
Write-Host $recoveryOutput -ForegroundColor DarkGray
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/catalog/status" -TimeoutSec 30
} catch {
    Write-Warning "The project catalog will refresh when the homepage is next opened: $($_.Exception.Message)"
}

Write-Host "" 
Write-Host "Cell Vision portable production is configured." -ForegroundColor Green
Write-Host "  Application: $applicationRoot"
Write-Host "  Workspace:   $workspaceRoot"
Write-Host "  Service:     CellVisionProduction (delayed automatic start)"
Write-Host "  URL:         http://127.0.0.1:$Port/"
