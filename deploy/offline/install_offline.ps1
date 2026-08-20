<# One-click installer executed from an unpacked offline release. #>
[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [string]$PackageRoot = "",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [int]$Port = 8777,
    [switch]$StartAfterInstall,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = $PSScriptRoot }
$packageRoot = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($PackageRoot))
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = if (Test-Path -LiteralPath "D:\" -PathType Container) {
        "D:\CellVision"
    } else {
        Join-Path $env:ProgramFiles "CellVision"
    }
}
$InstallRoot = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($InstallRoot))
$installVolumeRoot = [IO.Path]::GetPathRoot($InstallRoot)
if ([string]::Equals($InstallRoot, $installVolumeRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The installation directory cannot be a drive root. Choose a folder such as $installVolumeRoot`CellVision."
}
$releasePath = Join-Path $packageRoot "RELEASE.json"
$hashPath = Join-Path $packageRoot "SHA256SUMS.txt"
foreach ($required in @($releasePath, $hashPath, (Join-Path $packageRoot "Application"), (Join-Path $packageRoot "ModelBundle"), (Join-Path $packageRoot "wheelhouse"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Offline package is incomplete: $required" }
}
$release = Get-Content -LiteralPath $releasePath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $release.git_commit -or -not $release.git_clean) {
    throw "Release metadata does not contain a clean Git commit."
}

Write-Host "Verifying offline package integrity ..." -ForegroundColor Cyan
foreach ($line in Get-Content -LiteralPath $hashPath -Encoding UTF8) {
    if ($line -notmatch '^([0-9a-fA-F]{64})\s{2}(.+)$') { continue }
    $expected = $matches[1].ToLowerInvariant()
    $relative = $matches[2].Replace("/", "\")
    $target = [IO.Path]::GetFullPath((Join-Path $packageRoot $relative))
    if (-not $target.StartsWith($packageRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe path in SHA256SUMS.txt: $relative"
    }
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Package file missing: $relative" }
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Package integrity check failed: $relative" }
}
if ($VerifyOnly) {
    Write-Host "PACKAGE_VERIFICATION_OK" -ForegroundColor Green
    exit 0
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "The machine-wide Cell Vision installer requires administrator permission."
}

$installParent = Split-Path -Parent $InstallRoot
if ([string]::IsNullOrWhiteSpace($installParent)) {
    throw "Unable to determine the parent directory of the installation path: $InstallRoot"
}
if (-not (Test-Path -LiteralPath $installParent -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $installParent | Out-Null
}
$existingService = Get-Service -Name "CellVisionProduction" -ErrorAction SilentlyContinue
if ($null -ne $existingService) {
    Write-Host "Stopping the existing Cell Vision production service ..." -ForegroundColor DarkYellow
    if ($existingService.Status -ne "Stopped") {
        Stop-Service -Name "CellVisionProduction" -Force
        (Get-Service -Name "CellVisionProduction").WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
    }
    & sc.exe delete "CellVisionProduction" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Unable to remove the previous Cell Vision service." }
    Start-Sleep -Milliseconds 700
}
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    $commandLine = if ($null -ne $owner) { [string]$owner.CommandLine } else { "" }
    $isLegacyCellVision = $commandLine -match "cellvision" -and $commandLine -match "review-project"
    if (-not $isLegacyCellVision) {
        throw "Port $Port is used by another application (PID $($listener.OwningProcess))."
    }
    Write-Host "Stopping the previous user-session Cell Vision process (PID $($listener.OwningProcess)) ..." -ForegroundColor DarkYellow
    & taskkill.exe /PID $listener.OwningProcess /T /F | Out-Null
}
$staging = Join-Path $installParent ("CellVision.installing." + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $staging | Out-Null
try {
    Copy-Item -Path (Join-Path $packageRoot "Application\*") -Destination $staging -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $packageRoot "wheelhouse") -Destination (Join-Path $staging "wheelhouse") -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $packageRoot "ModelBundle") -Destination (Join-Path $staging "ModelBundle") -Recurse -Force
    Copy-Item -LiteralPath $releasePath -Destination (Join-Path $staging "RELEASE.json")

    if (Test-Path -LiteralPath $InstallRoot) {
        $backup = "$InstallRoot.backup-$(Get-Date -Format yyyyMMdd-HHmmss)"
        Write-Host "Moving the previous installation to: $backup" -ForegroundColor DarkYellow
        Move-Item -LiteralPath $InstallRoot -Destination $backup
    }
    Move-Item -LiteralPath $staging -Destination $InstallRoot
} finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}

$runtimeRoot = Join-Path $InstallRoot "Python312"
$python = Join-Path $runtimeRoot "python.exe"
function Test-BundledPythonRuntime {
    param([Parameter(Mandatory = $true)][string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { return $false }
    & $PythonPath -c "import encodings, pip, ssl, sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

if (-not (Test-BundledPythonRuntime -PythonPath $python)) {
    if (Test-Path -LiteralPath $runtimeRoot) {
        Write-Host "Removing an incomplete bundled Python runtime ..." -ForegroundColor DarkYellow
        Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
    }
    $installer = Get-ChildItem -LiteralPath (Join-Path $packageRoot "runtime") -Filter "python-3.12.*-amd64.exe" -File | Select-Object -First 1
    if ($null -eq $installer) { throw "Bundled Python 3.12 installer is missing." }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runtimeRoot) | Out-Null
    $installerLogRoot = Join-Path $env:ProgramData "CellVision\InstallerLogs"
    New-Item -ItemType Directory -Force -Path $installerLogRoot | Out-Null
    $pythonInstallerLog = Join-Path $installerLogRoot ("python-runtime-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
    Write-Host "Installing bundled Python runtime ..." -ForegroundColor Cyan
    $installerArguments = @(
        "/quiet", "InstallAllUsers=1", "PrependPath=0", "Include_launcher=0",
        "Include_test=0", "Include_doc=0", "Include_dev=0", "Include_tcltk=0",
        "Include_symbols=0", "Include_debug=0", "Include_pip=1",
        "TargetDir=$runtimeRoot", "/log", $pythonInstallerLog
    )
    & $installer.FullName @installerArguments
    $pythonInstallerExitCode = $LASTEXITCODE
    if ($pythonInstallerExitCode -ne 0) {
        throw "Python installer failed with exit code $pythonInstallerExitCode. Log: $pythonInstallerLog"
    }
}
if (-not (Test-BundledPythonRuntime -PythonPath $python)) {
    throw "Bundled Python runtime is incomplete after installation: $python"
}

$serviceRequirements = Join-Path $InstallRoot "requirements-windows-service.txt"
if (-not (Test-Path -LiteralPath $serviceRequirements -PathType Leaf)) {
    throw "Windows service requirements are missing: $serviceRequirements"
}
Write-Host "Installing the machine-wide Windows service host ..." -ForegroundColor Cyan
& $python -m pip install --no-index --find-links (Join-Path $InstallRoot "wheelhouse") -r $serviceRequirements
if ($LASTEXITCODE -ne 0) { throw "Unable to install the Windows service host." }
& $python -m pip install --no-index --find-links (Join-Path $InstallRoot "wheelhouse") --no-build-isolation --no-deps -e $InstallRoot
if ($LASTEXITCODE -ne 0) { throw "Unable to register Cell Vision in the service runtime." }
& $python -m pywin32_postinstall -install
if ($LASTEXITCODE -ne 0) { throw "Unable to finalize the Windows service runtime." }

$dataRoot = Join-Path $installParent "CellVisionData"
$stateRoot = Join-Path $installParent "CellVisionState"
$artifactRoot = Join-Path $stateRoot "artifacts"
$dbRoot = Join-Path $stateRoot "db"
$logRoot = Join-Path $stateRoot "logs"
$manifest = Join-Path $artifactRoot "projects\active\project.json"
$setup = Join-Path $InstallRoot "scripts\setup_production.ps1"
& $setup -InstallRoot $InstallRoot -DataRoot $dataRoot -ArtifactRoot $artifactRoot `
    -ModelRoot (Join-Path $InstallRoot "ModelBundle") -DbRoot $dbRoot -LogRoot $logRoot `
    -Manifest $manifest -BasePython $python -Wheelhouse (Join-Path $InstallRoot "wheelhouse") `
    -Offline -Device $Device -Port $Port
if ($LASTEXITCODE -ne 0) { throw "Cell Vision runtime setup failed with exit code $LASTEXITCODE." }

foreach ($sharedRoot in @($dataRoot, $artifactRoot)) {
    New-Item -ItemType Directory -Force -Path $sharedRoot | Out-Null
    & icacls.exe $sharedRoot /grant '*S-1-5-32-545:(OI)(CI)M' /T /C | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Unable to grant shared-user access: $sharedRoot" }
}
& icacls.exe $InstallRoot /grant '*S-1-5-32-545:(OI)(CI)RX' /T /C | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Unable to grant shared-user application access: $InstallRoot" }

$rootLauncherPath = Join-Path $InstallRoot "Start-CellVision.cmd"
$rootLauncherText = @'
@echo off
setlocal
title Cell Vision
powershell.exe -NoProfile -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:__PORT__/api/ready' -TimeoutSec 2; if ($r.status -eq 'ready') { exit 0 }; exit 1 } catch { exit 1 }"
if errorlevel 1 (
  echo.
  echo The Cell Vision machine service is not ready.
  echo Please ask an administrator to check the CellVisionProduction service.
  pause
  exit /b 1
)
"%~dp0.venv-production\Scripts\python.exe" -m cellvision.desktop_bridge --production-url "http://127.0.0.1:__PORT__/"
exit /b %ERRORLEVEL%
'@
$rootLauncherText = $rootLauncherText.Replace("__PORT__", [string]$Port)
[IO.File]::WriteAllText($rootLauncherPath, $rootLauncherText.TrimStart() + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))

$serviceInstaller = Join-Path $InstallRoot "scripts\install_production_service.ps1"
& $serviceInstaller -InstallRoot $InstallRoot -Port $Port -Action install
if ($LASTEXITCODE -ne 0) { throw "Unable to install the Cell Vision production service." }

$commonDesktop = [Environment]::GetFolderPath("CommonDesktopDirectory")
if (-not [string]::IsNullOrWhiteSpace($commonDesktop) -and (Test-Path -LiteralPath $commonDesktop -PathType Container)) {
    try {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut((Join-Path $commonDesktop "Cell Vision.lnk"))
        $shortcut.TargetPath = $rootLauncherPath
        $shortcut.Arguments = ""
        $shortcut.WorkingDirectory = $InstallRoot
        $shortcut.Description = "Open Cell Vision ($($release.git_short_commit))"
        $shortcut.Save()
    } catch {
        Write-Warning "The optional common-desktop shortcut could not be created: $($_.Exception.Message)"
    }
}

Write-Host "Installed Git commit: $($release.git_commit)" -ForegroundColor Green
Write-Host "Application: $InstallRoot" -ForegroundColor Green
Write-Host "Start: $rootLauncherPath" -ForegroundColor Green
Write-Host "Data import folder: $dataRoot" -ForegroundColor Green
if ($StartAfterInstall) {
    & $rootLauncherPath
    if ($LASTEXITCODE -ne 0) { throw "Cell Vision did not start successfully." }
}
