<#
.SYNOPSIS
    Wrap one Cell Vision offline ZIP as a single Windows installer EXE.

.DESCRIPTION
    Uses the Windows-built-in IExpress engine. The EXE extracts its embedded
    ZIP to a temporary directory and launches the existing graphical installer,
    so installation-path selection and progress reporting remain unchanged.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("production", "review")]
    [string]$Mode,
    [Parameter(Mandatory = $true)]
    [string]$SourceArchive,
    [string]$OutputPath = "",
    [switch]$Force,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

function ConvertTo-SedValue {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -match '[\r\n]') { throw "IExpress values cannot contain newlines." }
    return $Value.Replace('%', '%%')
}

$archivePath = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($SourceArchive))
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Source release archive was not found: $archivePath"
}
if ([IO.Path]::GetExtension($archivePath) -ine ".zip") {
    throw "Source release must be a ZIP archive: $archivePath"
}

$expectedLauncher = if ($Mode -eq "production") { "Install-CellVision.cmd" } else { "Install-CellVision-Review.cmd" }
$friendlyName = if ($Mode -eq "production") { "Cell Vision Production Installer" } else { "Cell Vision Review Installer" }
$outputStem = if ($Mode -eq "production") { "CellVision-Production-Setup.exe" } else { "CellVision-Review-Setup.exe" }
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path (Split-Path -Parent $archivePath) $outputStem
}
$exePath = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($OutputPath))
if ([IO.Path]::GetExtension($exePath) -ine ".exe") {
    throw "OutputPath must end in .exe: $exePath"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $matchingLaunchers = @($zip.Entries | Where-Object {
        $parts = @($_.FullName.Replace('\', '/').Trim('/').Split('/'))
        $parts.Count -le 2 -and $parts[-1].Equals($expectedLauncher, [StringComparison]::OrdinalIgnoreCase)
    })
    if ($matchingLaunchers.Count -ne 1) {
        throw "Release archive must contain exactly one $expectedLauncher; found $($matchingLaunchers.Count)."
    }
} finally {
    $zip.Dispose()
}

$iexpress = Join-Path $env:SystemRoot "System32\iexpress.exe"
if (-not (Test-Path -LiteralPath $iexpress -PathType Leaf)) {
    throw "Windows IExpress is unavailable: $iexpress"
}
if ($ValidateOnly) {
    Write-Host "SINGLE_FILE_INSTALLER_VALIDATION_OK: $Mode -> $exePath" -ForegroundColor Green
    exit 0
}

$outputParent = Split-Path -Parent $exePath
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null
if (Test-Path -LiteralPath $exePath) {
    if (-not $Force) { throw "Installer EXE already exists: $exePath. Use -Force to replace it." }
    [IO.File]::Delete($exePath)
}

$workingRoot = Join-Path ([IO.Path]::GetTempPath()) ("CellVisionIExpress-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $workingRoot | Out-Null
try {
    Copy-Item -LiteralPath $archivePath -Destination (Join-Path $workingRoot "payload.zip")

    $bootstrapPowerShell = @'
$ErrorActionPreference = "Stop"
$payload = Join-Path $PSScriptRoot "payload.zip"
$extractRoot = Join-Path ([IO.Path]::GetTempPath()) ("CellVisionInstaller-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $extractRoot | Out-Null
try {
    Expand-Archive -LiteralPath $payload -DestinationPath $extractRoot
    $launchers = @()
    $directLauncher = Join-Path $extractRoot "__LAUNCHER__"
    if (Test-Path -LiteralPath $directLauncher -PathType Leaf) {
        $launchers += Get-Item -LiteralPath $directLauncher
    }
    foreach ($packageRoot in Get-ChildItem -LiteralPath $extractRoot -Directory) {
        $candidate = Join-Path $packageRoot.FullName "__LAUNCHER__"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $launchers += Get-Item -LiteralPath $candidate
        }
    }
    if ($launchers.Count -ne 1) {
        throw "Embedded package must contain exactly one top-level __LAUNCHER__; found $($launchers.Count)."
    }
    Push-Location $launchers[0].DirectoryName
    try {
        & $launchers[0].FullName
        $installerExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($installerExitCode -ne 0) { exit $installerExitCode }
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        $_.Exception.Message,
        "Cell Vision Installer",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
    exit 1
} finally {
    if (Test-Path -LiteralPath $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit 0
'@
    $bootstrapPowerShell = $bootstrapPowerShell.Replace("__LAUNCHER__", $expectedLauncher)
    [IO.File]::WriteAllText(
        (Join-Path $workingRoot "bootstrap.ps1"),
        $bootstrapPowerShell.TrimStart() + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($true))
    )

    $bootstrapCmd = @'
@echo off
setlocal
powershell.exe -STA -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
exit /b %ERRORLEVEL%
'@
    [IO.File]::WriteAllText(
        (Join-Path $workingRoot "bootstrap.cmd"),
        $bootstrapCmd.TrimStart() + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )

    $sedPath = Join-Path $workingRoot "installer.sed"
    $sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3
[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=0
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=%InstallPrompt%
DisplayLicense=%DisplayLicense%
FinishMessage=%FinishMessage%
TargetName=%TargetName%
FriendlyName=%FriendlyName%
AppLaunched=%AppLaunched%
PostInstallCmd=%PostInstallCmd%
AdminQuietInstCmd=%AdminQuietInstCmd%
UserQuietInstCmd=%UserQuietInstCmd%
SourceFiles=SourceFiles
[Strings]
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=$(ConvertTo-SedValue -Value $exePath)
FriendlyName=$(ConvertTo-SedValue -Value $friendlyName)
AppLaunched=cmd.exe /d /c bootstrap.cmd
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
FILE0="bootstrap.cmd"
FILE1="bootstrap.ps1"
FILE2="payload.zip"
[SourceFiles]
SourceFiles0=$(ConvertTo-SedValue -Value ($workingRoot.TrimEnd('\') + '\'))
[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
"@
    [IO.File]::WriteAllText($sedPath, $sed.TrimStart(), [Text.Encoding]::Unicode)

    $iexpressProcess = Start-Process -FilePath $iexpress `
        -ArgumentList @("/N", $sedPath) -Wait -PassThru -WindowStyle Hidden
    if ($iexpressProcess.ExitCode -ne 0) {
        throw "IExpress failed with exit code $($iexpressProcess.ExitCode)."
    }
    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        throw "IExpress completed without creating the installer: $exePath"
    }
    Write-Host "Single-file installer ready: $exePath" -ForegroundColor Green
} finally {
    if (Test-Path -LiteralPath $workingRoot) {
        Remove-Item -LiteralPath $workingRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
