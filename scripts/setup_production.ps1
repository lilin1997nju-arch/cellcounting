<#
.SYNOPSIS
    Create the Windows inference-only environment for Cell Vision.

.DESCRIPTION
    The script creates a separate virtual environment, installs the runtime
    dependencies, probes the NVIDIA driver, installs a matching PyTorch wheel,
    writes .env.production, and checks the four model files required by the
    production worker. Training packages and pytest are not installed.

    With -Device auto (the default), CUDA is preferred only when nvidia-smi is
    usable and PyTorch subsequently confirms torch.cuda.is_available(). Any
    failed CUDA probe falls back to the CPU wheel automatically.
#>

[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [string]$DataRoot = "",
    [string]$ArtifactRoot = "",
    [string]$ModelRoot = "",
    [string]$DbRoot = "",
    [string]$LogRoot = "",
    [string]$Manifest = "",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [int]$Port = 8777,
    [string]$BindHost = "127.0.0.1",
    [switch]$AllowRemote,
    [switch]$SkipModelCheck,
    [switch]$ForceRecreate,
    [switch]$StartAfterSetup
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

function Convert-ToEnvPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    return $Value.Replace("\", "/")
}

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $script:PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Get-Python312 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        $candidate = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null | Select-Object -Last 1)
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            return $candidate.Trim()
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        $candidate = $python.Source
        $version = (& $candidate -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null | Select-Object -Last 1)
        if ($LASTEXITCODE -eq 0 -and $version) {
            $parts = $version.Trim().Split('.')
            if ($parts.Length -ge 2 -and [int]$parts[0] -eq 3 -and [int]$parts[1] -ge 11 -and [int]$parts[1] -le 13) {
                return $candidate
            }
        }
    }
    throw "Python 3.11-3.13 is required. Install Python 3.12 and ensure 'py -3.12' is available."
}

function Install-TorchWheel {
    param([Parameter(Mandatory = $true)][string]$IndexName)
    $indexUrl = "https://download.pytorch.org/whl/$IndexName"
    Write-Host "Installing PyTorch 2.11.0 / Torchvision 0.26.0 from $IndexName ..." -ForegroundColor Cyan
    & $script:PythonPath -m pip install --upgrade --force-reinstall --no-cache-dir --no-warn-script-location `
        --index-url $indexUrl --no-deps `
        "torch==2.11.0" "torchvision==0.26.0"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch installation failed for wheel index $IndexName."
    }
}

function Install-TorchSupport {
    # These are PyTorch runtime dependencies, not training dependencies. Keep
    # their source on PyPI because the PyTorch wheel indexes are not general
    # package indexes.
    & $script:PythonPath -m pip install --upgrade --no-cache-dir `
        "filelock>=3.13" "typing-extensions>=4.10" "sympy>=1.13.3" `
        "networkx>=2.5.1" "Jinja2>=3.1" "fsspec>=2023.5" "mpmath>=1.1.0"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch runtime support packages could not be installed."
    }
}

function Get-RuntimeInfo {
    param([ValidateSet("auto", "cpu", "cuda")][string]$RequestedDevice = "auto")
    $output = & $script:PythonPath -m cellvision runtime-info --device $RequestedDevice 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to probe the installed Cell Vision runtime: $($output -join "`n")"
    }
    return (($output -join "`n") | ConvertFrom-Json)
}

function Install-ProjectEditable {
    # Install the local package without resolving pyproject's unconstrained
    # torch requirement again. The selected wheel is installed explicitly.
    Invoke-Python @("-m", "pip", "install", "--no-deps", "--no-cache-dir", "-e", $InstallRoot)
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
        Write-Warning "Missing production model files:"
        $missing | ForEach-Object { Write-Warning "  $_" }
        return $false
    }
    return $true
}

$scriptRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = $scriptRoot
} else {
    $InstallRoot = Get-AbsolutePath -Value $InstallRoot -BasePath (Get-Location).Path
}

$parentRoot = Split-Path -Parent $InstallRoot
if ([string]::IsNullOrWhiteSpace($DataRoot)) { $DataRoot = Join-Path $parentRoot "cellvision-data" } else { $DataRoot = Get-AbsolutePath -Value $DataRoot -BasePath $InstallRoot }
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) { $ArtifactRoot = Join-Path $parentRoot "cellvision-artifacts" } else { $ArtifactRoot = Get-AbsolutePath -Value $ArtifactRoot -BasePath $InstallRoot }
if ([string]::IsNullOrWhiteSpace($ModelRoot)) { $ModelRoot = Join-Path $parentRoot "cellvision-models" } else { $ModelRoot = Get-AbsolutePath -Value $ModelRoot -BasePath $InstallRoot }
if ([string]::IsNullOrWhiteSpace($DbRoot)) { $DbRoot = Join-Path $parentRoot "cellvision-db" } else { $DbRoot = Get-AbsolutePath -Value $DbRoot -BasePath $InstallRoot }
if ([string]::IsNullOrWhiteSpace($LogRoot)) { $LogRoot = Join-Path $parentRoot "cellvision-logs" } else { $LogRoot = Get-AbsolutePath -Value $LogRoot -BasePath $InstallRoot }
if ([string]::IsNullOrWhiteSpace($Manifest)) { $Manifest = Join-Path $InstallRoot "artifacts\projects\active\project.json" } else { $Manifest = Get-AbsolutePath -Value $Manifest -BasePath $InstallRoot }

foreach ($requiredFile in @("requirements-production.txt", "pyproject.toml", "src", "review-ui")) {
    if (-not (Test-Path -LiteralPath (Join-Path $InstallRoot $requiredFile))) {
        throw "InstallRoot does not look like a Cell Vision application root: missing $requiredFile under $InstallRoot"
    }
}

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    throw "DataRoot could not be resolved."
}

foreach ($directory in @($DataRoot, $ArtifactRoot, $ModelRoot, $DbRoot, $LogRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$basePython = Get-Python312
$venvRoot = Join-Path $InstallRoot ".venv-production"
if ($ForceRecreate -and (Test-Path -LiteralPath $venvRoot)) {
    $resolvedVenv = [IO.Path]::GetFullPath($venvRoot)
    $resolvedInstall = [IO.Path]::GetFullPath($InstallRoot)
    if ($resolvedVenv -ne (Join-Path $resolvedInstall ".venv-production")) {
        throw "Refusing to recreate an unexpected virtual-environment path: $resolvedVenv"
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $venvRoot "Scripts\python.exe") -PathType Leaf)) {
    Write-Host "Creating production virtual environment: $venvRoot" -ForegroundColor Cyan
    & $basePython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Unable to create the production virtual environment." }
}
$script:PythonPath = Join-Path $venvRoot "Scripts\python.exe"

Write-Host "Installing base Python tooling ..." -ForegroundColor Cyan
Invoke-Python @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Write-Host "Installing production runtime dependencies ..." -ForegroundColor Cyan
Invoke-Python @("-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", (Join-Path $InstallRoot "requirements-production.txt"))

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
$nvidiaUsable = $false
if ($null -ne $nvidiaSmi) {
    $probe = & $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader 2>$null
    $nvidiaUsable = ($LASTEXITCODE -eq 0 -and @($probe).Count -gt 0)
}
$nvidiaAdapter = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "NVIDIA" } |
    Select-Object -First 1
$nvidiaPresent = $nvidiaUsable -or ($null -ne $nvidiaAdapter)
$tryCuda = ($Device -eq "cuda") -or ($Device -eq "auto" -and $nvidiaPresent)
$runtime = $null
$installedWheel = "cpu"

if ($tryCuda) {
    foreach ($cudaIndex in @("cu128", "cu126")) {
        try {
            Install-TorchWheel -IndexName $cudaIndex
            Install-TorchSupport
            Install-ProjectEditable
            $runtime = Get-RuntimeInfo -RequestedDevice $Device
            if ($runtime.selected_device -eq "cuda") {
                $installedWheel = $cudaIndex
                break
            }
            Write-Warning "PyTorch $cudaIndex was installed but CUDA is not usable; trying another wheel."
        } catch {
            Write-Warning $_.Exception.Message
        }
    }
}

if ($null -eq $runtime -or $runtime.selected_device -ne "cuda") {
    Install-TorchWheel -IndexName "cpu"
    Install-TorchSupport
    Install-ProjectEditable
    $runtime = Get-RuntimeInfo -RequestedDevice $Device
    $installedWheel = "cpu"
}

if ($runtime.selected_device -ne "cpu" -and $runtime.selected_device -ne "cuda") {
    throw "The installed runtime did not select CPU or CUDA: $($runtime | ConvertTo-Json -Compress)"
}

$modelBundleReady = Test-ModelBundle -Root $ModelRoot
if (-not $modelBundleReady -and -not $SkipModelCheck) {
    throw "Production model bundle is incomplete. Copy the four checkpoint files into $ModelRoot or rerun with -SkipModelCheck."
}

$manifestParent = Split-Path -Parent $Manifest
if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $manifestParent | Out-Null
    $scaffold = [ordered]@{
        project_id = "active"
        project_name = "Cell Vision Production"
        root = $DataRoot
        status = "ready"
        plates = @()
        created_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $scaffold | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Manifest -Encoding UTF8
    Write-Host "Created empty project manifest: $Manifest" -ForegroundColor DarkGray
}

$envPath = Join-Path $InstallRoot ".env.production"
if (Test-Path -LiteralPath $envPath -PathType Leaf) {
    $backupPath = "$envPath.bak-$(Get-Date -Format yyyyMMdd-HHmmss)"
    Copy-Item -LiteralPath $envPath -Destination $backupPath -Force
    Write-Host "Previous production environment backed up to $backupPath" -ForegroundColor DarkGray
}
$remoteValue = if ($AllowRemote) { "1" } else { "0" }
$envText = @"
# Generated by scripts/setup_production.ps1. Do not commit this file.
CELLVISION_PRODUCTION=1
CELLVISION_DATA_ROOT=$(Convert-ToEnvPath $DataRoot)
CELLVISION_ARTIFACT_ROOT=$(Convert-ToEnvPath $ArtifactRoot)
CELLVISION_MODEL_ROOT=$(Convert-ToEnvPath $ModelRoot)
CELLVISION_SHARED_MODEL_ROOT=$(Convert-ToEnvPath $ModelRoot)
CELLVISION_SOURCE_ARTIFACTS=$(Convert-ToEnvPath $ModelRoot)
CELLVISION_DB_ROOT=$(Convert-ToEnvPath $DbRoot)
CELLVISION_LOG_ROOT=$(Convert-ToEnvPath $LogRoot)
CELLVISION_MANIFEST=$(Convert-ToEnvPath $Manifest)
CELLVISION_HOST=$BindHost
CELLVISION_PORT=$Port
CELLVISION_ALLOW_REMOTE=$remoteValue
CELLVISION_DEVICE=$Device
CELLVISION_WORKER_DEVICE=$Device
CELLVISION_TORCH_WHEEL=$installedWheel
"@
[IO.File]::WriteAllText($envPath, $envText.TrimStart() + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))

Write-Host "" 
Write-Host "Cell Vision production environment is ready." -ForegroundColor Green
Write-Host "  Python:       $PythonPath"
Write-Host "  PyTorch wheel: $installedWheel"
Write-Host "  Runtime:      $($runtime.label)"
Write-Host "  Models:       $ModelRoot"
Write-Host "  Manifest:     $Manifest"
Write-Host "  Environment:  $envPath"
if ($runtime.selected_device -eq "cpu" -and $tryCuda) {
    Write-Warning "A CUDA-capable driver was probed, but PyTorch could not use it. Production will run on CPU."
}
if (-not $modelBundleReady) {
    Write-Warning "Model checks were skipped; start_production.ps1 will refuse to start until the model bundle is complete."
}

if ($StartAfterSetup) {
    & (Join-Path $InstallRoot "scripts\start_production.ps1") -InstallRoot $InstallRoot -Manifest $Manifest
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
