<# Build the Electron NSIS installer or extracted-folder ZIP from a portable runtime. #>
[CmdletBinding()]
param(
    [string]$PortableReleaseRoot = "",
    [switch]$SkipNpmInstall,
    [ValidateSet("nsis", "zip")][string]$PackageFormat = "nsis"
)

$ErrorActionPreference = "Stop"
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$electronRoot = Join-Path $repository "electron"
$payloadRoot = Join-Path $electronRoot "payload"
$payloadApplication = Join-Path $payloadRoot "Application"
$payloadFiles = Join-Path $payloadRoot "root"
$releaseContainer = Join-Path $repository "release"

if ([string]::IsNullOrWhiteSpace($PortableReleaseRoot)) {
    $candidate = Get-ChildItem -LiteralPath $releaseContainer -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "Application\Python312\python.exe") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $candidate) { throw "No portable release containing Application\Python312 was found." }
    $PortableReleaseRoot = $candidate.FullName
}
$portableRoot = [IO.Path]::GetFullPath($PortableReleaseRoot)
$portableApplication = Join-Path $portableRoot "Application"
foreach ($required in @(
    (Join-Path $portableApplication "Python312\python.exe"),
    (Join-Path $portableApplication "ModelBundle"),
    (Join-Path $electronRoot "package.json")
)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Installer input is incomplete: $required" }
}

$resolvedElectronRoot = [IO.Path]::GetFullPath($electronRoot).TrimEnd('\') + '\'
$resolvedPayload = [IO.Path]::GetFullPath($payloadRoot)
if (-not $resolvedPayload.StartsWith($resolvedElectronRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to rebuild payload outside the Electron workspace: $resolvedPayload"
}
if (Test-Path -LiteralPath $resolvedPayload) {
    Remove-Item -LiteralPath $resolvedPayload -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $payloadApplication, $payloadFiles | Out-Null

Write-Host "Copying the portable Python/model runtime ..." -ForegroundColor Cyan
Get-ChildItem -LiteralPath $portableApplication -Force | Copy-Item -Destination $payloadApplication -Recurse -Force

Write-Host "Overlaying the current source and review UI ..." -ForegroundColor Cyan
foreach ($overlay in @(
    @{ Source = (Join-Path $repository "src\cellvision"); Target = (Join-Path $payloadApplication "src\cellvision") },
    @{ Source = (Join-Path $repository "review-ui"); Target = (Join-Path $payloadApplication "review-ui") },
    @{ Source = (Join-Path $repository "scripts"); Target = (Join-Path $payloadApplication "scripts") }
)) {
    if (Test-Path -LiteralPath $overlay.Target) { Remove-Item -LiteralPath $overlay.Target -Recurse -Force }
    Copy-Item -LiteralPath $overlay.Source -Destination $overlay.Target -Recurse -Force
}

$resnetFilename = "resnet18-f37072fd.pth"
$resnetSha256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
$resnetSource = Join-Path $repository "artifacts\models\$resnetFilename"
if (-not (Test-Path -LiteralPath $resnetSource -PathType Leaf)) {
    throw "Required offline ResNet18 weight is missing: $resnetSource"
}
$actualResnetSha256 = (Get-FileHash -LiteralPath $resnetSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualResnetSha256 -ne $resnetSha256) {
    throw "Offline ResNet18 weight checksum mismatch: expected $resnetSha256, got $actualResnetSha256"
}
$resnetDestination = Join-Path $payloadApplication "ModelBundle\models\$resnetFilename"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resnetDestination) | Out-Null
Copy-Item -LiteralPath $resnetSource -Destination $resnetDestination -Force
$payloadPython = Join-Path $payloadApplication "Python312\python.exe"
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $payloadApplication "src"
    & $payloadPython -c (
        "import sys, torch; " +
        "from cellvision.pretrained import load_resnet18_imagenet_extractor; " +
        "model = load_resnet18_imagenet_extractor(sys.argv[1], torch.device('cpu')); " +
        "print(type(model).__name__)"
    ) $resnetDestination
    if ($LASTEXITCODE -ne 0) { throw "Bundled offline ResNet18 weight could not be loaded." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}

$portableFiles = @(
    "Configure-CellVision-Service.cmd",
    "Install-CellVision.cmd",
    "Uninstall-CellVision.cmd",
    "uninstall_cellvision.ps1",
    "uninstall_cellvision_cleanup.ps1",
    "install_cellvision_zip.ps1",
    "configure_service_launcher.ps1",
    "configure_service.ps1",
    "Start-CellVision.cmd",
    "start_cellvision_platform.ps1",
    "Recover-CellVision-Projects.cmd",
    "recover_cellvision_projects.ps1",
    "recover_cellvision_projects_launcher.ps1",
    "repair_project_manifests.py",
    "Import-CellVision-Workspace.cmd",
    "import_cellvision_workspace.py",
    "import_cellvision_workspace.ps1",
    "import_cellvision_workspace_launcher.ps1",
    "PROJECT-RECOVERY-README.txt",
    "PRODUCTION-MAINTENANCE-README.txt",
    "ZIP-INSTALL-README.txt"
)
foreach ($name in $portableFiles) {
    $source = Join-Path $repository "deploy\portable\$name"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing portable installer file: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $payloadFiles $name) -Force
}
$desktopVersion = [string](Get-Content -LiteralPath (Join-Path $electronRoot "package.json") `
    -Raw -Encoding UTF8 | ConvertFrom-Json).version
[IO.File]::WriteAllText(
    (Join-Path $payloadFiles "CELLVISION_DESKTOP_VERSION.txt"),
    ($desktopVersion + [Environment]::NewLine),
    (New-Object Text.UTF8Encoding($false))
)

$cacheRoot = Join-Path $repository "artifacts\cache"
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
$redistributable = Join-Path $cacheRoot "vc_redist.x64.exe"
if (-not (Test-Path -LiteralPath $redistributable -PathType Leaf)) {
    Write-Host "Downloading the official Microsoft Visual C++ x64 runtime ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $redistributable -UseBasicParsing
}
$signature = Get-AuthenticodeSignature -LiteralPath $redistributable
if ($signature.Status -ne "Valid" -or [string]$signature.SignerCertificate.Subject -notmatch "Microsoft") {
    throw "The cached Visual C++ runtime does not have a valid Microsoft signature."
}
Copy-Item -LiteralPath $redistributable -Destination (Join-Path $payloadFiles "vc_redist.x64.exe") -Force

if (-not $SkipNpmInstall) {
    Write-Host "Restoring pinned Electron build dependencies ..." -ForegroundColor Cyan
    & npm.cmd ci --prefix $electronRoot
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
}

if ($PackageFormat -eq "nsis") {
    Write-Host "Building the x64 NSIS installer ..." -ForegroundColor Cyan
    Push-Location $electronRoot
    try {
        & npm.cmd run pack:win
        if ($LASTEXITCODE -ne 0) { throw "Electron installer build failed." }
    } finally {
        Pop-Location
    }
    $installer = Get-ChildItem -LiteralPath (Join-Path $electronRoot "dist") `
        -Filter "CellVision-Setup-*-x64.exe" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $installer) { throw "Electron builder completed without an installer artifact." }
    Write-Host "Electron installer ready: $($installer.FullName)" -ForegroundColor Green
    exit 0
}

Write-Host "Building the extracted-folder Electron layout ..." -ForegroundColor Cyan
Push-Location $electronRoot
try {
    & npm.cmd run pack:dir
    if ($LASTEXITCODE -ne 0) { throw "Electron directory build failed." }
} finally {
    Pop-Location
}

$unpackedRoot = Join-Path $electronRoot "dist\win-unpacked"
foreach ($required in @(
    (Join-Path $unpackedRoot "Cell Vision.exe"),
    (Join-Path $unpackedRoot "Install-CellVision.cmd"),
    (Join-Path $unpackedRoot "Uninstall-CellVision.cmd"),
    (Join-Path $unpackedRoot "uninstall_cellvision.ps1"),
    (Join-Path $unpackedRoot "uninstall_cellvision_cleanup.ps1"),
    (Join-Path $unpackedRoot "CELLVISION_DESKTOP_VERSION.txt"),
    (Join-Path $unpackedRoot "install_cellvision_zip.ps1"),
    (Join-Path $unpackedRoot "Application\Python312\python.exe"),
    (Join-Path $unpackedRoot "Application\ModelBundle\models\resnet18-f37072fd.pth")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Extracted-folder build is incomplete: $required"
    }
}

$manifestPath = Join-Path $unpackedRoot "CELLVISION_PACKAGE_FILES.txt"
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) { Remove-Item -LiteralPath $manifestPath -Force }
$manifestEntries = Get-ChildItem -LiteralPath $unpackedRoot -File -Recurse -Force |
    ForEach-Object {
        $_.FullName.Substring($unpackedRoot.Length).TrimStart("\").Replace("\", "/")
    } |
    Where-Object { $_ -notmatch '^(?i:Workspace)/' } |
    Sort-Object -Unique
[IO.File]::WriteAllLines($manifestPath, $manifestEntries, (New-Object Text.UTF8Encoding($false)))

$zipPath = Join-Path $electronRoot "dist\CellVision-Desktop-$desktopVersion-x64.zip"
if (Test-Path -LiteralPath $zipPath -PathType Leaf) { Remove-Item -LiteralPath $zipPath -Force }

$sevenZip = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source
if ([string]::IsNullOrWhiteSpace($sevenZip)) {
    $sevenZip = Get-ChildItem -LiteralPath (Join-Path $env:LOCALAPPDATA "electron-builder\Cache") `
        -Filter "7za.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if ([string]::IsNullOrWhiteSpace($sevenZip) -or -not (Test-Path -LiteralPath $sevenZip -PathType Leaf)) {
    throw "7-Zip command line runtime was not found in PATH or the electron-builder cache."
}

Write-Host "Creating a ZIP that extracts directly into the selected installation folder ..." -ForegroundColor Cyan
Push-Location $unpackedRoot
try {
    & $sevenZip a -tzip -mx=5 -mmt=on $zipPath ".\*"
    if ($LASTEXITCODE -ne 0) { throw "ZIP creation failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
& $sevenZip t $zipPath
if ($LASTEXITCODE -ne 0) { throw "ZIP integrity verification failed with exit code $LASTEXITCODE." }

$zip = Get-Item -LiteralPath $zipPath
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Cell Vision extracted-folder ZIP ready: $($zip.FullName)" -ForegroundColor Green
Write-Host "  Bytes:   $($zip.Length)"
Write-Host "  SHA-256: $hash"
