<# Request elevation for project metadata backup/recovery when installed under Program Files. #>
[CmdletBinding()]
param([int]$StartupTimeoutSeconds = 180)

$ErrorActionPreference = "Stop"
$recoveryScript = Join-Path $PSScriptRoot "recover_cellvision_projects.ps1"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    & $recoveryScript -StartupTimeoutSeconds $StartupTimeoutSeconds
    exit $LASTEXITCODE
}
$escapedScript = $recoveryScript.Replace("'", "''")
$command = "& '$escapedScript' -StartupTimeoutSeconds $StartupTimeoutSeconds"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList @(
    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded
) -WorkingDirectory $PSScriptRoot -Wait -PassThru
exit $process.ExitCode
