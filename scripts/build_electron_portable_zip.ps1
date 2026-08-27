<# Build the production Electron ZIP that is extracted directly into its final folder. #>
[CmdletBinding()]
param(
    [string]$PortableReleaseRoot = "",
    [switch]$SkipNpmInstall
)

$ErrorActionPreference = "Stop"
$builder = Join-Path $PSScriptRoot "build_electron_installer.ps1"
$builderParameters = @{ PackageFormat = "zip" }
if (-not [string]::IsNullOrWhiteSpace($PortableReleaseRoot)) {
    $builderParameters.PortableReleaseRoot = $PortableReleaseRoot
}
if ($SkipNpmInstall) { $builderParameters.SkipNpmInstall = $true }
& $builder @builderParameters
exit $LASTEXITCODE
