<#
.SYNOPSIS
    Build a Git-pinned, network-free Cell Vision Windows release.

.DESCRIPTION
    Run this script on a connected Windows build PC. It archives the exact
    committed Git tree, copies the three production checkpoints, downloads a
    Python 3.12 installer and a complete Windows wheelhouse, then emits one ZIP.
    When a compatible previous release exists, its runtime and wheelhouse are
    reused so application-only changes do not contact the network.
    The target PC only needs Windows and enough disk space; installation never
    contacts the network.
#>

[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [ValidateSet("cpu", "cu128")]
    [string]$TorchVariant = "cpu",
    [string]$PythonVersion = "3.12.10",
    [string]$BuilderPython = "",
    [string]$ReuseAssetsFrom = "",
    [switch]$NoAssetReuse,
    [switch]$KeepDirectory
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string]$FilePath, [Parameter(Mandatory = $true)][string[]]$Arguments)
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Get-RelativeReleasePath {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $rootPrefix = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Release file is outside the release root: $fullPath"
    }
    return $fullPath.Substring($rootPrefix.Length)
}

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
        return -join @($algorithm.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") })
    } finally {
        $algorithm.Dispose()
    }
}

function Get-AssetContract {
    param(
        [Parameter(Mandatory = $true)][string]$ApplicationRoot,
        [Parameter(Mandatory = $true)][string]$RequestedPython,
        [Parameter(Mandatory = $true)][string]$RequestedVariant
    )
    $productionRequirements = Join-Path $ApplicationRoot "requirements-production.txt"
    $reviewRequirements = Join-Path $ApplicationRoot "requirements-portable-review.txt"
    if (-not (Test-Path -LiteralPath $productionRequirements -PathType Leaf)) { return "" }
    if (-not (Test-Path -LiteralPath $reviewRequirements -PathType Leaf)) { return "" }
    $contract = @(
        "python=$RequestedPython",
        "platform=win_amd64-cp312",
        "torch_variant=$RequestedVariant",
        "torch=2.11.0",
        "torchvision=0.26.0",
        "bootstrap=pip,setuptools,wheel,filelock,typing-extensions,sympy,networkx,Jinja2,fsspec,mpmath",
        (Get-Content -LiteralPath $productionRequirements -Raw),
        (Get-Content -LiteralPath $reviewRequirements -Raw)
    ) -join "`n---`n"
    return Get-TextSha256 -Value $contract
}

$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$git = (Get-Command git -ErrorAction Stop).Source
$dirty = @(& $git -C $repository status --porcelain)
if ($dirty.Count -gt 0) {
    throw "Release builds require a clean Git worktree. Commit the current changes first."
}
$commit = (& $git -C $repository rev-parse HEAD).Trim()
$shortCommit = (& $git -C $repository rev-parse --short=10 HEAD).Trim()
if ([string]::IsNullOrWhiteSpace($commit)) { throw "Unable to resolve the release Git commit." }

if ([string]::IsNullOrWhiteSpace($BuilderPython)) {
    $venvPython = Join-Path $repository ".venv\Scripts\python.exe"
    $BuilderPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { (Get-Command python -ErrorAction Stop).Source }
}
$BuilderPython = [IO.Path]::GetFullPath($BuilderPython)

$releaseParent = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $repository "release"
} else {
    [IO.Path]::GetFullPath($OutputDirectory)
}
New-Item -ItemType Directory -Force -Path $releaseParent | Out-Null
$releaseName = "CellVision-offline-$shortCommit-$TorchVariant"
$releaseRoot = Join-Path $releaseParent $releaseName
if (Test-Path -LiteralPath $releaseRoot) {
    throw "Release directory already exists: $releaseRoot"
}
$zipPath = "$releaseRoot.zip"
if (Test-Path -LiteralPath $zipPath) {
    throw "Release ZIP already exists: $zipPath"
}

$currentAssetContract = Get-AssetContract `
    -ApplicationRoot $repository `
    -RequestedPython $PythonVersion `
    -RequestedVariant $TorchVariant
$assetSource = $null
if (-not $NoAssetReuse) {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($ReuseAssetsFrom)) {
        $candidates = @([IO.Path]::GetFullPath($ReuseAssetsFrom))
    } else {
        $candidates = @(
            Get-ChildItem -LiteralPath $releaseParent -Directory -Filter "CellVision-offline-*-$TorchVariant" |
                Where-Object { $_.FullName -ne $releaseRoot } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -ExpandProperty FullName
        )
    }
    foreach ($candidate in $candidates) {
        $candidateRelease = Join-Path $candidate "RELEASE.json"
        $candidateApplication = Join-Path $candidate "Application"
        $candidateWheelhouse = Join-Path $candidate "wheelhouse"
        $candidateInstaller = Join-Path $candidate "runtime\python-$PythonVersion-amd64.exe"
        if (-not (Test-Path -LiteralPath $candidateRelease -PathType Leaf)) { continue }
        if (-not (Test-Path -LiteralPath $candidateWheelhouse -PathType Container)) { continue }
        if (-not (Test-Path -LiteralPath $candidateInstaller -PathType Leaf)) { continue }
        if ((Get-Item -LiteralPath $candidateInstaller).Length -le 0) { continue }
        try {
            $candidateMetadata = Get-Content -LiteralPath $candidateRelease -Raw | ConvertFrom-Json
        } catch {
            continue
        }
        if ([string]$candidateMetadata.python_version -ne $PythonVersion) { continue }
        if ([string]$candidateMetadata.torch_variant -ne $TorchVariant) { continue }
        if ([string]$candidateMetadata.torch_version -ne "2.11.0") { continue }
        if ([string]$candidateMetadata.torchvision_version -ne "0.26.0") { continue }
        $candidateContract = Get-AssetContract `
            -ApplicationRoot $candidateApplication `
            -RequestedPython $PythonVersion `
            -RequestedVariant $TorchVariant
        if ([string]::IsNullOrWhiteSpace($candidateContract) -or $candidateContract -ne $currentAssetContract) { continue }
        if (@(Get-ChildItem -LiteralPath $candidateWheelhouse -File).Count -eq 0) { continue }
        $assetSource = $candidate
        break
    }
}

$application = Join-Path $releaseRoot "Application"
$wheelhouse = Join-Path $releaseRoot "wheelhouse"
$runtime = Join-Path $releaseRoot "runtime"
$modelBundle = Join-Path $releaseRoot "ModelBundle"
foreach ($directory in @($application, $wheelhouse, $runtime, $modelBundle)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

try {
    $archive = Join-Path $releaseRoot ".application.zip"
    Invoke-Checked $git @("-C", $repository, "archive", "--format=zip", "--output=$archive", $commit)
    Expand-Archive -LiteralPath $archive -DestinationPath $application
    Remove-Item -LiteralPath $archive -Force
    [IO.File]::WriteAllText(
        (Join-Path $application "RELEASE_GIT_COMMIT.txt"),
        $commit + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )

    $models = @(
        @{ Source = Join-Path $repository "artifacts\models\teaching_classifier.pt"; Destination = "models\teaching_classifier.pt" },
        @{ Source = Join-Path $repository "artifacts\models\multiplicity_classifier.pt"; Destination = "models\multiplicity_classifier.pt" },
        @{ Source = Join-Path $repository "artifacts\v2\models\latest_instance_segmenter.pt"; Destination = "v2\models\latest_instance_segmenter.pt" }
    )
    foreach ($model in $models) {
        if (-not (Test-Path -LiteralPath $model.Source -PathType Leaf)) {
            throw "Required production checkpoint is missing: $($model.Source)"
        }
        $destination = Join-Path $modelBundle $model.Destination
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $model.Source -Destination $destination
    }

    Copy-Item -LiteralPath (Join-Path $repository "deploy\offline\install_offline.ps1") -Destination (Join-Path $releaseRoot "install_offline.ps1")
    Copy-Item -LiteralPath (Join-Path $repository "deploy\offline\Install-CellVision.cmd") -Destination (Join-Path $releaseRoot "Install-CellVision.cmd")
    Copy-Item -LiteralPath (Join-Path $repository "deploy\windows\install_ui.ps1") -Destination (Join-Path $releaseRoot "install_ui.ps1")

    $pythonInstaller = Join-Path $runtime "python-$PythonVersion-amd64.exe"
    if ($null -ne $assetSource) {
        Write-Host "Reusing offline runtime assets from: $assetSource" -ForegroundColor Cyan
        Copy-Item -Path (Join-Path $assetSource "runtime\*") -Destination $runtime -Recurse -Force
        Copy-Item -Path (Join-Path $assetSource "wheelhouse\*") -Destination $wheelhouse -Recurse -Force
    } else {
        $pythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
        Write-Host "Downloading Python $PythonVersion installer ..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller -UseBasicParsing

        $targetArgs = @(
            "-m", "pip", "download", "--dest", $wheelhouse,
            "--only-binary=:all:", "--platform", "win_amd64", "--python-version", "312",
            "--implementation", "cp", "--abi", "cp312"
        )
        Write-Host "Downloading production dependency wheelhouse ..." -ForegroundColor Cyan
        Invoke-Checked $BuilderPython ($targetArgs + @(
            "-r", (Join-Path $repository "requirements-production.txt"),
            "pip", "setuptools", "wheel", "filelock", "typing-extensions", "sympy", "networkx", "Jinja2", "fsspec", "mpmath"
        ))
        $torchIndex = "https://download.pytorch.org/whl/$TorchVariant"
        Write-Host "Downloading PyTorch $TorchVariant wheelhouse ..." -ForegroundColor Cyan
        Invoke-Checked $BuilderPython ($targetArgs + @(
            "--index-url", $torchIndex, "--extra-index-url", "https://pypi.org/simple",
            "torch==2.11.0", "torchvision==0.26.0"
        ))
    }

    $modelMetadata = foreach ($model in $models) {
        $packaged = Join-Path $modelBundle $model.Destination
        [ordered]@{
            path = $model.Destination.Replace("\", "/")
            bytes = (Get-Item -LiteralPath $packaged).Length
            sha256 = (Get-FileHash -LiteralPath $packaged -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $release = [ordered]@{
        product = "Cell Vision"
        release_format = 1
        git_commit = $commit
        git_short_commit = $shortCommit
        git_clean = $true
        built_at = (Get-Date).ToUniversalTime().ToString("o")
        python_version = $PythonVersion
        torch_version = "2.11.0"
        torchvision_version = "0.26.0"
        torch_variant = $TorchVariant
        asset_contract_sha256 = $currentAssetContract
        runtime_assets_reused_from = if ($null -ne $assetSource) {
            (Get-Content -LiteralPath (Join-Path $assetSource "RELEASE.json") -Raw | ConvertFrom-Json).git_commit
        } else { "" }
        production_mode = $true
        training_surfaces_included = $false
        models = @($modelMetadata)
    }
    $release | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $releaseRoot "RELEASE.json") -Encoding UTF8

    $hashLines = Get-ChildItem -LiteralPath $releaseRoot -Recurse -File |
        Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = (Get-RelativeReleasePath -Root $releaseRoot -Path $_.FullName).Replace("\", "/")
            "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $relative"
        }
    [IO.File]::WriteAllLines((Join-Path $releaseRoot "SHA256SUMS.txt"), $hashLines, (New-Object Text.UTF8Encoding($false)))

    Write-Host "Compressing offline release ..." -ForegroundColor Cyan
    Compress-Archive -LiteralPath $releaseRoot -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host "Offline release ready: $zipPath" -ForegroundColor Green
} catch {
    Write-Error $_
    throw
} finally {
    if (-not $KeepDirectory -and (Test-Path -LiteralPath $releaseRoot) -and (Test-Path -LiteralPath $zipPath)) {
        Remove-Item -LiteralPath $releaseRoot -Recurse -Force
    }
}
