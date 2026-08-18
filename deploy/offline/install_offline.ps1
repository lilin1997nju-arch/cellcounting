<# One-click installer executed from an unpacked offline release. #>
[CmdletBinding()]
param(
    [string]$InstallRoot = "",
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [int]$Port = 8777,
    [switch]$StartAfterInstall
)

$ErrorActionPreference = "Stop"
$packageRoot = [IO.Path]::GetFullPath($PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Join-Path $env:LOCALAPPDATA "CellVision"
}
$InstallRoot = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($InstallRoot))
$releasePath = Join-Path $packageRoot "RELEASE.json"
$hashPath = Join-Path $packageRoot "SHA256SUMS.txt"
foreach ($required in @($releasePath, $hashPath, (Join-Path $packageRoot "Application"), (Join-Path $packageRoot "ModelBundle"), (Join-Path $packageRoot "wheelhouse"))) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Offline package is incomplete: $required" }
}
$release = Get-Content -LiteralPath $releasePath -Raw | ConvertFrom-Json
if (-not $release.git_commit -or -not $release.git_clean) {
    throw "Release metadata does not contain a clean Git commit."
}

Write-Host "Verifying offline package integrity ..." -ForegroundColor Cyan
foreach ($line in Get-Content -LiteralPath $hashPath) {
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

$installParent = Split-Path -Parent $InstallRoot
New-Item -ItemType Directory -Force -Path $installParent | Out-Null
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

$runtimeRoot = Join-Path $env:LOCALAPPDATA "CellVisionRuntime\Python312"
$python = Join-Path $runtimeRoot "python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $installer = Get-ChildItem -LiteralPath (Join-Path $packageRoot "runtime") -Filter "python-3.12.*-amd64.exe" -File | Select-Object -First 1
    if ($null -eq $installer) { throw "Bundled Python 3.12 installer is missing." }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runtimeRoot) | Out-Null
    Write-Host "Installing bundled Python runtime ..." -ForegroundColor Cyan
    $process = Start-Process -FilePath $installer.FullName -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
        "Include_test=0", "Include_doc=0", "Include_pip=1", "TargetDir=$runtimeRoot"
    ) -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Python installer failed with exit code $($process.ExitCode)." }
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Bundled Python runtime was not installed: $python" }

$dataRoot = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "CellVisionData"
$stateRoot = Join-Path $env:LOCALAPPDATA "CellVisionState"
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

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $desktop "Cell Vision.lnk"))
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $InstallRoot 'scripts\start_production.ps1')`" -InstallRoot `"$InstallRoot`""
$shortcut.WorkingDirectory = $InstallRoot
$shortcut.Description = "Start Cell Vision production service ($($release.git_short_commit))"
$shortcut.Save()

Write-Host "Installed Git commit: $($release.git_commit)" -ForegroundColor Green
Write-Host "Application: $InstallRoot" -ForegroundColor Green
Write-Host "Data import folder: $dataRoot" -ForegroundColor Green
if ($StartAfterInstall) {
    & (Join-Path $InstallRoot "scripts\start_production.ps1") -InstallRoot $InstallRoot
    if ($LASTEXITCODE -ne 0) { throw "Cell Vision did not start successfully." }
    Start-Process "http://127.0.0.1:$Port/"
}
