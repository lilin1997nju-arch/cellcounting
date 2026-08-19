[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"
function ConvertTo-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + $Value.Replace('"', '\"') + '"'
}
function Test-Python312 {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $version = (& $Path -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
        return ($LASTEXITCODE -eq 0 -and ([string]$version).Trim() -eq "3.12")
    } catch {
        return $false
    }
}
function Find-Python312 {
    param([string]$Preferred = "")
    $candidates = [Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($Preferred)) { $candidates.Add($Preferred) }
    foreach ($registryPath in @(
        "Registry::HKEY_CURRENT_USER\Software\Python\PythonCore\3.12\InstallPath",
        "Registry::HKEY_LOCAL_MACHINE\Software\Python\PythonCore\3.12\InstallPath",
        "Registry::HKEY_LOCAL_MACHINE\Software\WOW6432Node\Python\PythonCore\3.12\InstallPath"
    )) {
        if (Test-Path -LiteralPath $registryPath) {
            $registeredRoot = (Get-Item -LiteralPath $registryPath).GetValue("")
            if (-not [string]::IsNullOrWhiteSpace($registeredRoot)) {
                $candidates.Add((Join-Path $registeredRoot "python.exe"))
            }
        }
    }
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
        $candidates.Add($command.Source)
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Python312 -Path $candidate) { return [IO.Path]::GetFullPath($candidate) }
    }
    return $null
}
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
$preferredPython = Join-Path $pythonRoot "python.exe"
$basePython = Find-Python312 -Preferred $preferredPython
if ([string]::IsNullOrWhiteSpace($basePython)) {
    $installer = Get-ChildItem -LiteralPath $runtime -Filter "python-3.12.*-amd64.exe" -File | Select-Object -First 1
    if ($null -eq $installer) { throw "Bundled Python 3.12 installer is missing." }
    Write-Host "Installing the lightweight review runtime ..." -ForegroundColor Cyan
    $arguments = @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
        "Include_test=0", "Include_doc=0", "Include_pip=1", "TargetDir=$pythonRoot"
    )
    $quotedArguments = @($arguments | ForEach-Object { ConvertTo-ProcessArgument -Value ([string]$_) })
    $process = Start-Process -FilePath $installer.FullName -ArgumentList $quotedArguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Python installer failed: $($process.ExitCode)" }
    $basePython = Find-Python312 -Preferred $preferredPython
    if ([string]::IsNullOrWhiteSpace($basePython)) {
        throw "Python 3.12 installation completed but python.exe could not be located."
    }
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
