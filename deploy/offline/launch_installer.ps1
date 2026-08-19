<# Elevate the production installer once and wait for it to finish. #>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ui = Join-Path $PSScriptRoot "install_ui.ps1"
if (-not (Test-Path -LiteralPath $ui -PathType Leaf)) {
    throw "Cell Vision installer UI is missing: $ui"
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$arguments = @(
    "-STA", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", ('"' + $ui + '"'), "-Mode", "production"
)
if ($isAdministrator) {
    & powershell.exe @arguments
    exit $LASTEXITCODE
}
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -Verb RunAs -Wait -PassThru
exit $process.ExitCode
