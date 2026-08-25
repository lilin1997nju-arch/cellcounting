<# Build the one-file Electron/NSIS production installer from a portable runtime. #>
[CmdletBinding()]
param(
    [string]$PortableReleaseRoot = "",
    [switch]$SkipNpmInstall
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

$portableFiles = @(
    "Configure-CellVision-Service.cmd",
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
    "PRODUCTION-MAINTENANCE-README.txt"
)
foreach ($name in $portableFiles) {
    $source = Join-Path $repository "deploy\portable\$name"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing portable installer file: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $payloadFiles $name) -Force
}

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

Write-Host "Building the x64 NSIS installer ..." -ForegroundColor Cyan
Push-Location $electronRoot
try {
    & npm.cmd run pack:win
    if ($LASTEXITCODE -ne 0) { throw "Electron installer build failed." }
} finally {
    Pop-Location
}
$installer = Get-ChildItem -LiteralPath (Join-Path $electronRoot "dist") -Filter "CellVision-Setup-*-x64.exe" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $installer) { throw "Electron builder completed without an installer artifact." }
Write-Host "Electron installer ready: $($installer.FullName)" -ForegroundColor Green
