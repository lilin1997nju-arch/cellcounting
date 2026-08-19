[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"
$bundleRoot = if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "Application") -PathType Container) {
    [IO.Path]::GetFullPath($PSScriptRoot)
} else {
    [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
}
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Join-Path $env:LOCALAPPDATA "CellVisionReviewPlatform"
}
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$sourceApplication = Join-Path $bundleRoot "Application"
$sourceWheelhouse = Join-Path $bundleRoot "wheelhouse"
$sourceRuntime = Join-Path $bundleRoot "runtime"
foreach ($required in @($sourceApplication, $sourceWheelhouse, $sourceRuntime)) {
    if (-not (Test-Path -LiteralPath $required -PathType Container)) {
        throw "Review platform bundle is incomplete: $required"
    }
}

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$application = Join-Path $InstallRoot "Application"
$wheelhouse = Join-Path $InstallRoot "wheelhouse"
$runtime = Join-Path $InstallRoot "runtime"
foreach ($pair in @(
    @($sourceApplication, $application),
    @($sourceWheelhouse, $wheelhouse),
    @($sourceRuntime, $runtime)
)) {
    New-Item -ItemType Directory -Force -Path $pair[1] | Out-Null
    Copy-Item -Path (Join-Path $pair[0] "*") -Destination $pair[1] -Recurse -Force
}

$pythonRoot = Join-Path $InstallRoot "Python312"
$basePython = Join-Path $pythonRoot "python.exe"
if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
    $installer = Get-ChildItem -LiteralPath $runtime -Filter "python-3.12.*-amd64.exe" -File | Select-Object -First 1
    if ($null -eq $installer) { throw "Bundled Python 3.12 installer is missing." }
    Write-Host "Installing the lightweight review runtime ..." -ForegroundColor Cyan
    $arguments = @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
        "Include_test=0", "Include_doc=0", "Include_pip=1", "TargetDir=$pythonRoot"
    )
    $process = Start-Process -FilePath $installer.FullName -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Python installer failed: $($process.ExitCode)" }
}

$venvRoot = Join-Path $InstallRoot ".venv-review-platform"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $basePython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Unable to create review platform environment." }
}
& $venvPython -m pip install --no-index --find-links $wheelhouse --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Unable to install base Python tooling." }
& $venvPython -m pip install --no-index --find-links $wheelhouse -r (Join-Path $application "requirements-portable-review.txt")
if ($LASTEXITCODE -ne 0) { throw "Unable to install review dependencies." }
& $venvPython -m pip install --no-index --find-links $wheelhouse --no-build-isolation --no-deps -e $application
if ($LASTEXITCODE -ne 0) { throw "Unable to install Cell Vision review application." }

$launcherSource = Join-Path $application "deploy\review_platform\windows"
foreach ($name in @("Open-CellVision-Review.cmd", "open_review_platform.ps1")) {
    Copy-Item -LiteralPath (Join-Path $launcherSource $name) -Destination (Join-Path $InstallRoot $name) -Force
}

if (-not $NoDesktopShortcut) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "Cell Vision Review.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = Join-Path $InstallRoot "Open-CellVision-Review.cmd"
    $shortcut.WorkingDirectory = $InstallRoot
    $shortcut.Description = "Import and review Cell Vision .cvreview data packages"
    $shortcut.Save()
}

Write-Host "Cell Vision Review Platform is ready: $InstallRoot" -ForegroundColor Green
Write-Host "Open: $(Join-Path $InstallRoot 'Open-CellVision-Review.cmd')"
