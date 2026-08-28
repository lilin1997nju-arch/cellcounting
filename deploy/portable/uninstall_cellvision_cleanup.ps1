<# Delayed exact-file cleanup used after Uninstall-CellVision.cmd exits. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DeploymentRoot,
    [Parameter(Mandatory = $true)][string]$ManifestPath,
    [Parameter(Mandatory = $true)][string]$LogPath
)

$ErrorActionPreference = "Stop"
Start-Sleep -Seconds 3
$root = [IO.Path]::GetFullPath($DeploymentRoot).TrimEnd("\")
$rootPrefix = $root + "\"
$driveRoot = [IO.Path]::GetPathRoot($root).TrimEnd("\")
$workspace = [IO.Path]::GetFullPath((Join-Path $root "Workspace")).TrimEnd("\")
$workspacePrefix = $workspace + "\"
if ([string]::IsNullOrWhiteSpace($root) -or $root -eq $driveRoot) {
    throw "Refusing delayed cleanup at drive root: $root"
}
if (-not (Test-Path -LiteralPath $workspace -PathType Container)) {
    throw "Preserved Workspace is missing; refusing program cleanup: $workspace"
}
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Uninstall manifest is missing: $ManifestPath"
}

function Write-CleanupLog {
    param([Parameter(Mandatory = $true)][string]$Message)
    Add-Content -LiteralPath $LogPath -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message) -Encoding UTF8
}

try {
    Write-CleanupLog "Delayed program-file cleanup started for $root"
    $application = [IO.Path]::GetFullPath((Join-Path $root "Application"))
    if (-not $application.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        $application.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Application cleanup target failed validation: $application"
    }
    if (Test-Path -LiteralPath $application) {
        Remove-Item -LiteralPath $application -Recurse -Force
    }

    foreach ($relative in Get-Content -LiteralPath $ManifestPath -Encoding UTF8) {
        $entry = [string]$relative
        if ([string]::IsNullOrWhiteSpace($entry)) { continue }
        $normalized = $entry.Replace("/", "\").TrimStart("\")
        if ([IO.Path]::IsPathRooted($entry) -or $normalized -match '(^|\\)\.\.(\\|$)') {
            throw "Unsafe uninstall manifest entry: $entry"
        }
        $target = [IO.Path]::GetFullPath((Join-Path $root $normalized))
        if (-not $target.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Uninstall target escaped the installation root: $target"
        }
        if ($target -eq $workspace -or $target.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        }
    }
    foreach ($generated in @("CELLVISION_ZIP_INSTALL.json", "CELLVISION_PACKAGE_FILES.txt")) {
        $target = [IO.Path]::GetFullPath((Join-Path $root $generated))
        if ($target.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        }
    }
    Get-ChildItem -LiteralPath $root -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -ne $workspace -and
            -not $_.FullName.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)
        } |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
            if ($null -eq (Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue | Select-Object -First 1)) {
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
            }
        }
    Write-CleanupLog "Program files removed successfully. Workspace preserved at $workspace"
} catch {
    Write-CleanupLog "ERROR: $($_.Exception.Message)"
} finally {
    Remove-Item -LiteralPath $ManifestPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
}
