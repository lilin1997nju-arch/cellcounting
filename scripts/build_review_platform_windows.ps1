[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceRelease,
    [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$source = [IO.Path]::GetFullPath($SourceRelease)
$commit = (Get-Content -LiteralPath (Join-Path $source "Application\RELEASE_GIT_COMMIT.txt") -Raw).Trim()
if ([string]::IsNullOrWhiteSpace($commit)) { throw "Source release has no Git commit." }
$short = $commit.Substring(0, [Math]::Min(10, $commit.Length))
$parent = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $repository "release"
} else {
    [IO.Path]::GetFullPath($OutputDirectory)
}
$target = Join-Path $parent "CellVisionReviewPlatform-$short-windows-x64"
$zip = "$target.zip"
if (Test-Path -LiteralPath $target) { throw "Target already exists: $target" }
if (Test-Path -LiteralPath $zip) { throw "Target already exists: $zip" }
New-Item -ItemType Directory -Force -Path $target | Out-Null

try {
    $application = Join-Path $target "Application"
    $archive = Join-Path $target ".application.zip"
    & git -C $repository archive --format=zip --output=$archive $commit
    if ($LASTEXITCODE -ne 0) { throw "Unable to archive application Git commit." }
    Expand-Archive -LiteralPath $archive -DestinationPath $application
    Remove-Item -LiteralPath $archive -Force
    [IO.File]::WriteAllText(
        (Join-Path $application "RELEASE_GIT_COMMIT.txt"),
        $commit + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )

    Copy-Item -LiteralPath (Join-Path $source "runtime") -Destination (Join-Path $target "runtime") -Recurse
    $wheelhouse = Join-Path $target "wheelhouse"
    New-Item -ItemType Directory -Force -Path $wheelhouse | Out-Null
    $excluded = @("torch-", "torchvision-", "sympy-", "mpmath-", "fsspec-", "filelock-")
    Get-ChildItem -LiteralPath (Join-Path $source "wheelhouse") -File | ForEach-Object {
        $lower = $_.Name.ToLowerInvariant()
        $skip = $false
        foreach ($prefix in $excluded) { if ($lower.StartsWith($prefix)) { $skip = $true; break } }
        if (-not $skip) { Copy-Item -LiteralPath $_.FullName -Destination $wheelhouse }
    }
    $deploy = Join-Path $application "deploy\review_platform\windows"
    foreach ($name in @("Install-CellVision-Review.cmd", "install_review_platform.ps1")) {
        Copy-Item -LiteralPath (Join-Path $deploy $name) -Destination (Join-Path $target $name)
    }
    [ordered]@{
        product = "Cell Vision Review Platform"
        platform = "windows-x64"
        git_commit = $commit
        data_format = "cellvision-review-data"
        data_format_version = 1
        model_runtime_included = $false
        built_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $target "RELEASE.json") -Encoding UTF8
    Compress-Archive -LiteralPath $target -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Review platform ready: $zip" -ForegroundColor Green
} catch {
    Write-Error $_
    throw
}
