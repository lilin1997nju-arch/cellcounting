<# Start a copyable task workspace with the exact production review UI. #>
[CmdletBinding()]
param([int]$Port = 8788)

$ErrorActionPreference = "Stop"

function ConvertTo-StartProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + $Value.Replace('"', '\"') + '"'
}

$reviewRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$application = Join-Path $reviewRoot "application"
$projectManifest = Join-Path $reviewRoot "project\project.json"
$package = Join-Path $reviewRoot "PACKAGE.json"
$wheelhouse = Join-Path $reviewRoot "wheelhouse"
$runtimeBundle = Join-Path $reviewRoot "runtime"
foreach ($required in @($application, $projectManifest, $package, $wheelhouse, $runtimeBundle)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Offline review workspace is incomplete: $required" }
}

$pythonRoot = Join-Path $reviewRoot "Python312"
$basePython = Join-Path $pythonRoot "python.exe"
if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
    $installer = Get-ChildItem -LiteralPath $runtimeBundle -Filter "python-3.12.*-amd64.exe" -File | Select-Object -First 1
    if ($null -eq $installer) { throw "Bundled Python installer is missing." }
    Write-Host "Installing the bundled offline review runtime ..." -ForegroundColor Cyan
    $process = Start-Process -FilePath $installer.FullName -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=0", "Include_launcher=0",
        "Include_test=0", "Include_doc=0", "Include_pip=1", "TargetDir=$pythonRoot"
    ) -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) { throw "Python installer failed with exit code $($process.ExitCode)." }
}

& $basePython (Join-Path $application "scripts\rebase_portable_review.py") --package-root $reviewRoot
if ($LASTEXITCODE -ne 0) { throw "Unable to rebase the copied task paths." }

$venvRoot = Join-Path $reviewRoot ".venv-review"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Preparing the offline review environment (first launch only) ..." -ForegroundColor Cyan
    & $basePython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Unable to create the review environment." }
    & $venvPython -m pip install --no-index --find-links $wheelhouse --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Unable to install base Python tooling." }
    & $venvPython -m pip install --no-index --find-links $wheelhouse -r (Join-Path $application "requirements-production.txt")
    if ($LASTEXITCODE -ne 0) { throw "Unable to install review dependencies." }
    & $venvPython -m pip install --no-index --find-links $wheelhouse torch torchvision
    if ($LASTEXITCODE -ne 0) { throw "Unable to install the image runtime." }
    & $venvPython -m pip install --no-index --find-links $wheelhouse --no-build-isolation --no-deps -e $application
    if ($LASTEXITCODE -ne 0) { throw "Unable to install Cell Vision review code." }
}

$metadata = Get-Content -LiteralPath $package -Raw | ConvertFrom-Json
$projectId = [Uri]::EscapeDataString([string]$metadata.project_id)
$env:CELLVISION_PRODUCTION = "1"
$env:CELLVISION_PORTABLE_REVIEW = "1"
$env:CELLVISION_DEVICE = "cpu"
$env:CELLVISION_WORKER_DEVICE = "cpu"

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    try {
        $runningProject = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/project" -TimeoutSec 3
        if ([string]$runningProject.project_id -ne [string]$metadata.project_id -or -not [bool]$runningProject.portable_review) {
            throw "Port $Port is already used by another service. Close it or run start_offline_review.ps1 -Port <another port>."
        }
    } catch {
        throw "Port $Port is already used by another service. Close it or run start_offline_review.ps1 -Port <another port>."
    }
} else {
    $logRoot = Join-Path $reviewRoot "logs"
    New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
    $arguments = @(
        "-m", "cellvision", "review-project", "--manifest", $projectManifest,
        "--host", "127.0.0.1", "--port", [string]$Port, "--no-worker"
    )
    $quotedArguments = @($arguments | ForEach-Object { ConvertTo-StartProcessArgument -Value ([string]$_) })
    $process = Start-Process -FilePath $venvPython -ArgumentList $quotedArguments -WorkingDirectory $application `
        -RedirectStandardOutput (Join-Path $logRoot "review.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "review.stderr.log") `
        -WindowStyle Hidden -PassThru
    $ready = $false
    for ($attempt = 1; $attempt -le 60; $attempt++) {
        try {
            $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/ready" -TimeoutSec 3
            if ($response.status -eq "ready") { $ready = $true; break }
        } catch {
            if ($process.HasExited) { break }
        }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { throw "The offline review service did not become ready. See $logRoot." }
}

Start-Process "http://127.0.0.1:$Port/projects/$projectId/"
