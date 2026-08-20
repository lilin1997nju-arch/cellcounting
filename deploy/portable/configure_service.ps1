<# Configure the copied portable production folder as a machine-wide service. #>
[CmdletBinding()]
param(
    [int]$Port = 8777,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$python = Join-Path $applicationRoot "Python312\python.exe"
$serviceHost = Join-Path $applicationRoot "Python312\pythonservice.exe"
$serviceRuntimeDll = Get-ChildItem -LiteralPath (Join-Path $applicationRoot "Python312") `
    -Filter "pywintypes*.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1

foreach ($required in @(
    $python,
    $serviceHost,
    (Join-Path $applicationRoot "src\cellvision"),
    (Join-Path $applicationRoot "ModelBundle"),
    (Join-Path $applicationRoot "scripts\setup_production.ps1")
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

Write-Host "Configuring shared domain-user permissions ..." -ForegroundColor Cyan
& icacls.exe $applicationRoot /inheritance:e /grant `
    '*S-1-5-32-545:(OI)(CI)RX' '*S-1-5-11:(OI)(CI)RX' /C /Q | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Unable to grant shared application access." }
& icacls.exe $workspaceRoot /inheritance:e /grant `
    '*S-1-5-32-545:(OI)(CI)M' '*S-1-5-11:(OI)(CI)M' /C /Q | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Unable to grant shared workspace access." }

Write-Host "Registering and starting Cell Vision service; readiness can take up to 90 seconds ..." -ForegroundColor Cyan
$serviceInstaller = Join-Path $applicationRoot "scripts\install_production_service.ps1"
& $serviceInstaller -InstallRoot $applicationRoot -Port $Port -Action install
if ($LASTEXITCODE -ne 0) { throw "Unable to install the Cell Vision production service." }

Write-Host "" 
Write-Host "Cell Vision portable production is configured." -ForegroundColor Green
Write-Host "  Application: $applicationRoot"
Write-Host "  Workspace:   $workspaceRoot"
Write-Host "  Service:     CellVisionProduction (delayed automatic start)"
Write-Host "  URL:         http://127.0.0.1:$Port/"
