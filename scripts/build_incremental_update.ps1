<# Build the one-click Cell Vision production update package. #>
[CmdletBinding()]
param(
    [string]$PackageId = "cellvision-update-20260824-r1",
    [string]$TargetVersion = "2026.08.24-r1",
    [string]$ProductionBaseline = "CellVision-offline-6231fdce81-cpu (informational only)",
    [string]$ProductionApplicationRoot = "",
    [string]$OutputRoot = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if ([string]::IsNullOrWhiteSpace($ProductionApplicationRoot)) {
    $ProductionApplicationRoot = Join-Path $repositoryRoot "release\CellVision-offline-6231fdce81-cpu\Application"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repositoryRoot "deploy\updates"
}
$ProductionApplicationRoot = [IO.Path]::GetFullPath($ProductionApplicationRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$packageName = "CellVision-Update-20260824-r1"
$packageRoot = [IO.Path]::GetFullPath((Join-Path $OutputRoot $packageName))
$zipPath = [IO.Path]::GetFullPath((Join-Path $OutputRoot "$packageName.zip"))
$zipHashPath = "$zipPath.sha256.txt"

function Assert-DirectChild {
    param([string]$Root, [string]$Path)
    $normalizedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($normalizedRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify a path outside the update output root: $resolved"
    }
}

$runtimeFiles = @(
    "review-ui/auto-review.css",
    "review-ui/auto-review.html",
    "review-ui/auto-review.js",
    "review-ui/offline-review.html",
    "review-ui/offline-review.js",
    "review-ui/project-dashboard.css",
    "review-ui/project-dashboard.html",
    "review-ui/project-dashboard.js",
    "src/cellvision/offline_review.py",
    "src/cellvision/project_server.py",
    "src/cellvision/review_payloads.py",
    "src/cellvision/review_quick_review.py",
    "src/cellvision/review_screening_api.py",
    "src/cellvision/review_summary.py",
    "src/cellvision/well_screening.py"
)

foreach ($relative in $runtimeFiles) {
    $source = Join-Path $repositoryRoot $relative
    $productionTarget = Join-Path $ProductionApplicationRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Source file is missing: $relative" }
    if (-not (Test-Path -LiteralPath $productionTarget -PathType Leaf)) { throw "Production target file is missing: $relative" }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Assert-DirectChild -Root $OutputRoot -Path $packageRoot
Assert-DirectChild -Root $OutputRoot -Path $zipPath
Assert-DirectChild -Root $OutputRoot -Path $zipHashPath
foreach ($existingPath in @($packageRoot, $zipPath, $zipHashPath)) {
    if (Test-Path -LiteralPath $existingPath) {
        if (-not $Force) { throw "Update output already exists: $existingPath (pass -Force to rebuild)" }
        Remove-Item -LiteralPath $existingPath -Recurse -Force
    }
}

$payloadRoot = Join-Path $packageRoot "payload"
New-Item -ItemType Directory -Force -Path $payloadRoot | Out-Null
foreach ($relative in $runtimeFiles) {
    $source = Join-Path $repositoryRoot $relative
    $destination = Join-Path $payloadRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

Copy-Item -LiteralPath (Join-Path $repositoryRoot "deploy\windows\Update-CellVision.cmd") -Destination (Join-Path $packageRoot "Update-CellVision.cmd")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "deploy\windows\update_incremental.ps1") -Destination (Join-Path $packageRoot "update.ps1")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "deploy\windows\Rollback-CellVision.cmd") -Destination (Join-Path $packageRoot "Rollback-CellVision.cmd")
Copy-Item -LiteralPath (Join-Path $repositoryRoot "deploy\windows\rollback_incremental.ps1") -Destination (Join-Path $packageRoot "rollback.ps1")

$manifest = [ordered]@{
    format = "cellvision-incremental-update"
    version = 1
    package_id = $PackageId
    target_version = $TargetVersion
    production_baseline = $ProductionBaseline
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    files = @($runtimeFiles | ForEach-Object { [ordered]@{ path = $_ } })
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $packageRoot "UPDATE.json") -Encoding UTF8

$readmeLines = @(
    "Cell Vision Production Update $TargetVersion",
    "========================================",
    "",
    "Install:",
    "1. Put the entire $packageName folder in the Cell Vision installation root.",
    "2. Double-click Update-CellVision.cmd and approve the Windows administrator prompt.",
    "3. The updater locates CellVisionProduction, backs up 15 files, replaces them, restarts the service, and checks /api/ready.",
    "4. Installation is complete when update completed successfully is shown.",
    "",
    "Rollback:",
    "1. Keep this update folder and the update-backups folder inside the application directory.",
    "2. Double-click Rollback-CellVision.cmd to restore the latest update backup and restart the service.",
    "",
    "Notes:",
    "- The updater does not validate a production baseline or individual file hashes.",
    "- Every replaced file is backed up first; a failed update restores this backup automatically.",
    "- Workspace, data, models, Python runtime, and production configuration are not replaced.",
    "- Deferred cell-multiplicity and debris model changes are not included."
)
$readme = $readmeLines -join [Environment]::NewLine
$readme | Set-Content -LiteralPath (Join-Path $packageRoot "README.txt") -Encoding UTF8

Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText($zipHashPath, "$zipHash  $packageName.zip", [Text.Encoding]::ASCII)
Write-Host "Update folder: $packageRoot" -ForegroundColor Green
Write-Host "Update archive: $zipPath" -ForegroundColor Green
Write-Host "Archive SHA256: $zipHash"
