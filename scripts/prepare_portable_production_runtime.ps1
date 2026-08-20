<# Build a relocatable, MSI-free Python runtime inside a release Application folder. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ApplicationRoot,
    [Parameter(Mandatory = $true)][string]$Wheelhouse,
    [string]$PythonVersion = "3.12.10"
)

$ErrorActionPreference = "Stop"
$application = [IO.Path]::GetFullPath($ApplicationRoot)
$wheels = [IO.Path]::GetFullPath($Wheelhouse)
if (-not (Test-Path -LiteralPath $application -PathType Container)) { throw "Application root not found: $application" }
if (-not (Test-Path -LiteralPath $wheels -PathType Container)) { throw "Wheelhouse not found: $wheels" }
$runtime = Join-Path $application "Python312"
if (Test-Path -LiteralPath $runtime) { throw "Portable runtime already exists: $runtime" }

$working = Join-Path ([IO.Path]::GetTempPath()) ("CellVisionPython-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $working | Out-Null
try {
    $package = Join-Path $working "python.$PythonVersion.nupkg"
    $expanded = Join-Path $working "expanded"
    $url = "https://www.nuget.org/api/v2/package/python/$PythonVersion"
    Write-Host "Downloading portable Python $PythonVersion ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $package -UseBasicParsing
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($package, $expanded)
    Copy-Item -LiteralPath (Join-Path $expanded "tools") -Destination $runtime -Recurse

    $sitePackages = Join-Path $runtime "Lib\site-packages"
    New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null
    [IO.File]::WriteAllText(
        (Join-Path $sitePackages "cellvision-portable.pth"),
        "../../../src" + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )
    $python = Join-Path $runtime "python.exe"
    Write-Host "Installing the prebuilt production runtime ..." -ForegroundColor Cyan
    & $python -m pip install --disable-pip-version-check --no-cache-dir --no-index `
        --find-links $wheels --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Unable to prepare portable Python tooling." }
    & $python -m pip install --disable-pip-version-check --no-cache-dir --no-index `
        --find-links $wheels -r (Join-Path $application "requirements-production.txt") `
        -r (Join-Path $application "requirements-windows-service.txt") `
        "torch==2.11.0" "torchvision==0.26.0"
    if ($LASTEXITCODE -ne 0) { throw "Unable to prepare portable production dependencies." }
    & $python -c "import cellvision, fastapi, torch, win32serviceutil; print(torch.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "Portable production runtime verification failed." }

    # Keep the pywin32 service host self-contained. The traditional
    # pywin32_postinstall step writes into Windows system directories and is
    # intentionally not used by this relocatable deployment.
    $serviceHostSource = Join-Path $sitePackages "win32\pythonservice.exe"
    $serviceRuntimeSource = Get-ChildItem -LiteralPath (Join-Path $sitePackages "pywin32_system32") `
        -Filter "pywintypes*.dll" -File | Select-Object -First 1
    if (-not (Test-Path -LiteralPath $serviceHostSource -PathType Leaf) -or $null -eq $serviceRuntimeSource) {
        throw "The portable pywin32 service files are incomplete."
    }
    Copy-Item -LiteralPath $serviceHostSource -Destination (Join-Path $runtime "pythonservice.exe") -Force
    Copy-Item -LiteralPath $serviceRuntimeSource.FullName -Destination (Join-Path $runtime $serviceRuntimeSource.Name) -Force
    Write-Host "PORTABLE_PRODUCTION_RUNTIME_READY=$runtime" -ForegroundColor Green
} catch {
    if (Test-Path -LiteralPath $runtime) { Remove-Item -LiteralPath $runtime -Recurse -Force }
    throw
} finally {
    if (Test-Path -LiteralPath $working) { Remove-Item -LiteralPath $working -Recurse -Force }
}
