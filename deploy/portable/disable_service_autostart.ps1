<# Keep the Cell Vision service installed but prevent it from starting at boot. #>
[CmdletBinding()]
param([switch]$ValidateOnly)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionProduction"
if ($ValidateOnly) {
    Write-Output "DISABLE_AUTOSTART_SCRIPT_VALIDATION_OK"
    exit 0
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Administrator permission is required to change the Cell Vision service startup mode."
}

$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
if ($null -eq $service) {
    Write-Host "Cell Vision service is not installed; no automatic startup is configured." -ForegroundColor DarkYellow
    exit 0
}

& sc.exe config $serviceName start= demand | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Unable to change the Cell Vision service startup mode." }
$serviceInfo = Get-CimInstance Win32_Service -Filter "Name='$serviceName'"
if ($null -eq $serviceInfo -or [string]$serviceInfo.StartMode -ne "Manual") {
    throw "Cell Vision service startup mode was not changed to Manual."
}

Write-Host "Cell Vision automatic startup is disabled." -ForegroundColor Green
Write-Host "The currently running service was not stopped. It will not start automatically after the next reboot."
