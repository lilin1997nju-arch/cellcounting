<# Configure the copied portable production folder as a machine-wide service. #>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "cpu",
    [switch]$RepairService,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionDesktopProduction"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$python = Join-Path $applicationRoot "Python312\python.exe"
$serviceHost = Join-Path $applicationRoot "Python312\pythonservice.exe"
$recoveryScript = Join-Path $deploymentRoot "repair_project_manifests.py"
$environmentPath = Join-Path $applicationRoot ".env.production"
$instanceIdPath = Join-Path $workspaceRoot ".cellvision-instance-id"
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

Write-Host "Checking the bundled Python executable ..." -ForegroundColor Cyan
& $python -I -c "import sys; print(sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) { throw "The bundled Python executable could not start." }
if ($ValidateOnly) {
    Write-Host "Running the full Cell Vision/PyTorch validation ..." -ForegroundColor Cyan
    & $python -u -c "import cellvision, fastapi, torch, win32serviceutil; print(torch.__version__, flush=True)"
    if ($LASTEXITCODE -ne 0) { throw "The portable Python runtime is incomplete." }
    Write-Output "PORTABLE_SERVICE_CONFIGURATION_VALIDATION_OK"
    exit 0
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Administrator permission is required to configure the machine-wide Cell Vision service."
}

function Get-EnvironmentValue {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) { return "" }
    $prefix = $Name + "="
    $line = Get-Content -LiteralPath $environmentPath | Where-Object {
        $_.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -First 1
    if ($null -eq $line) { return "" }
    return $line.Substring($prefix.Length).Trim()
}

function Test-TcpPortAvailable {
    param([Parameter(Mandatory = $true)][int]$Candidate)
    $listener = $null
    try {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Candidate)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) { try { $listener.Stop() } catch {} }
    }
}

function Find-AvailablePort {
    param([Parameter(Mandatory = $true)][int]$Preferred)
    for ($offset = 1; $offset -le 1000; $offset++) {
        $candidate = $Preferred + $offset
        if ($candidate -gt 65535) { break }
        if (Test-TcpPortAvailable -Candidate $candidate) { return $candidate }
    }
    throw "No free local TCP port was found after port $Preferred."
}

$projectsRoot = Join-Path $workspaceRoot "Projects"
$databaseRoot = Join-Path $workspaceRoot "Database"
$logsRoot = Join-Path $workspaceRoot "Logs"
$inboxRoot = Join-Path $workspaceRoot "Inbox"
$manifest = Join-Path $projectsRoot "active\project.json"
foreach ($directory in @($workspaceRoot, $projectsRoot, $databaseRoot, $logsRoot, $inboxRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}
if (-not (Test-Path -LiteralPath $instanceIdPath -PathType Leaf)) {
    [IO.File]::WriteAllText(
        $instanceIdPath,
        ([guid]::NewGuid().ToString("D") + [Environment]::NewLine),
        (New-Object Text.UTF8Encoding($false))
    )
}
$instanceId = (Get-Content -LiteralPath $instanceIdPath -Raw -Encoding UTF8).Trim()
if ([string]::IsNullOrWhiteSpace($instanceId)) { throw "Cell Vision instance ID is empty: $instanceIdPath" }

if ($Port -le 0) {
    $configuredPort = Get-EnvironmentValue -Name "CELLVISION_PORT"
    if ($configuredPort -match '^\d+$' -and [int]$configuredPort -gt 0) {
        $Port = [int]$configuredPort
    } else {
        $Port = 8777
    }
}

$existingServiceBeforeSetup = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
$portMayBelongToCurrentService = $false
if ($null -ne $existingServiceBeforeSetup -and $existingServiceBeforeSetup.Status -ne "Stopped") {
    $serviceInfoBeforeSetup = Get-CimInstance Win32_Service -Filter "Name='$serviceName'" -ErrorAction Stop
    $pathNameBeforeSetup = [string]$serviceInfoBeforeSetup.PathName
    $actualHostBeforeSetup = ""
    if ($pathNameBeforeSetup -match '^\s*"([^"]+)"') { $actualHostBeforeSetup = $Matches[1] }
    elseif ($pathNameBeforeSetup -match '^\s*([^\s]+)') { $actualHostBeforeSetup = $Matches[1] }
    $portMayBelongToCurrentService = -not [string]::IsNullOrWhiteSpace($actualHostBeforeSetup) -and [string]::Equals(
        [IO.Path]::GetFullPath($actualHostBeforeSetup),
        [IO.Path]::GetFullPath($serviceHost),
        [StringComparison]::OrdinalIgnoreCase
    )
}
if (-not $portMayBelongToCurrentService -and -not (Test-TcpPortAvailable -Candidate $Port)) {
    $occupiedPort = $Port
    $Port = Find-AvailablePort -Preferred $occupiedPort
    Write-Warning "Port $occupiedPort is occupied by another process. This installation will use port $Port."
}

Write-Host "Backing up project metadata before service configuration ..." -ForegroundColor Cyan
$backupOutput = & $python $recoveryScript --projects-root $projectsRoot --backup
if ($LASTEXITCODE -ne 0) { throw "Unable to back up project metadata before service configuration." }
Write-Host $backupOutput -ForegroundColor DarkGray

$setup = Join-Path $applicationRoot "scripts\setup_production.ps1"
Write-Host "Preparing the bundled CPU runtime; GPU/WMI detection is disabled for this package ..." -ForegroundColor Cyan
$previousPipNoIndex = $env:PIP_NO_INDEX
$previousPipDisableVersionCheck = $env:PIP_DISABLE_PIP_VERSION_CHECK
try {
    $env:PIP_NO_INDEX = "1"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    & $setup -InstallRoot $applicationRoot -DataRoot $inboxRoot -ArtifactRoot $workspaceRoot `
        -ModelRoot (Join-Path $applicationRoot "ModelBundle") -DbRoot $databaseRoot `
        -LogRoot $logsRoot -Manifest $manifest -BasePython $python `
        -UseBasePythonRuntime -SkipDependencyInstall -Device $Device -Port $Port `
        -InstanceId $instanceId -ServiceName $serviceName
    if ($LASTEXITCODE -ne 0) { throw "Cell Vision portable runtime configuration failed." }
} finally {
    $env:PIP_NO_INDEX = $previousPipNoIndex
    $env:PIP_DISABLE_PIP_VERSION_CHECK = $previousPipDisableVersionCheck
}

Write-Host "Using the existing administrator and LocalSystem permissions; no recursive ACL rewrite is needed." -ForegroundColor DarkGray
Write-Host "Ensuring the Cell Vision service is registered and ready ..." -ForegroundColor Cyan
$serviceInstaller = Join-Path $applicationRoot "scripts\install_production_service.ps1"
$legacyService = Get-Service -Name "CellVisionProduction" -ErrorAction SilentlyContinue
if ($null -ne $legacyService) {
    $legacyInfo = Get-CimInstance Win32_Service -Filter "Name='CellVisionProduction'" -ErrorAction Stop
    $legacyPathName = [string]$legacyInfo.PathName
    $legacyHost = ""
    if ($legacyPathName -match '^\s*"([^"]+)"') { $legacyHost = $Matches[1] }
    elseif ($legacyPathName -match '^\s*([^\s]+)') { $legacyHost = $Matches[1] }
    $legacyUsesThisInstall = -not [string]::IsNullOrWhiteSpace($legacyHost) -and [string]::Equals(
        [IO.Path]::GetFullPath($legacyHost),
        [IO.Path]::GetFullPath($serviceHost),
        [StringComparison]::OrdinalIgnoreCase
    )
    if ($legacyUsesThisInstall) {
        Write-Host "Migrating this installation from CellVisionProduction to $serviceName ..." -ForegroundColor Cyan
        & $serviceInstaller -InstallRoot $applicationRoot -ServiceName "CellVisionProduction" -Action uninstall
        if ($LASTEXITCODE -ne 0) { throw "Unable to remove the legacy service for this installation." }
    }
}
$existingService = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
$installService = $null -eq $existingService
if ($null -ne $existingService) {
    $serviceInfo = Get-CimInstance Win32_Service -Filter "Name='$serviceName'" -ErrorAction Stop
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
            "$serviceName belongs to a different installation: $actualHost. " +
            "Refusing to switch Workspace automatically. Use the daily launcher from that installation, " +
            "or rerun configure_service.ps1 with -RepairService after confirming the intended folder."
        )
    }
    $installService = -not $sameHost
}
if ($installService) {
    & $serviceInstaller -InstallRoot $applicationRoot -Port $Port -ServiceName $serviceName -Action install
    if ($LASTEXITCODE -ne 0) { throw "Unable to install the Cell Vision production service." }
} else {
    if ($existingService.Status -ne "Stopped") {
        & $serviceInstaller -InstallRoot $applicationRoot -Port $Port -ServiceName $serviceName -Action stop
        if ($LASTEXITCODE -ne 0) { throw "Unable to stop the existing Cell Vision production service." }
    }
    & $serviceInstaller -InstallRoot $applicationRoot -Port $Port -ServiceName $serviceName -Action start
    if ($LASTEXITCODE -ne 0) { throw "Unable to start the existing Cell Vision production service." }
    $ready = $false
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ready" -TimeoutSec 2
            if ($response.status -eq "ready" -and [string]$response.instance_id -eq $instanceId) {
                $ready = $true
                break
            }
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
Write-Host "  Service:     $serviceName (delayed automatic start)"
Write-Host "  URL:         http://127.0.0.1:$Port/"
