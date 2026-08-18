<#
.SYNOPSIS
    Build a Git-pinned, network-free Cell Vision Windows release.

.DESCRIPTION
    Run this script on a connected Windows build PC. It archives the exact
    committed Git tree, copies the three production checkpoints, downloads a
    Python 3.12 installer and a complete Windows wheelhouse, then emits one ZIP.
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

    $pythonInstaller = Join-Path $runtime "python-$PythonVersion-amd64.exe"
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
        production_mode = $true
        training_surfaces_included = $false
        models = @($modelMetadata)
    }
    $release | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $releaseRoot "RELEASE.json") -Encoding UTF8

    $hashLines = Get-ChildItem -LiteralPath $releaseRoot -Recurse -File |
        Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($releaseRoot, $_.FullName).Replace("\", "/")
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
