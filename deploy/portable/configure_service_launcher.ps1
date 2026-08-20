[CmdletBinding()]
param(
    [ValidateSet("configure_service.ps1", "disable_service_autostart.ps1")]
    [string]$ScriptName = "configure_service.ps1"
)

$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot $ScriptName
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "Portable service configuration script is missing: $target"
}
$logRoot = Join-Path $env:LOCALAPPDATA "CellVisionInstallerLogs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logRoot ("portable-{0}-{1}.log" -f `
    [IO.Path]::GetFileNameWithoutExtension($ScriptName), $stamp)
$literalTarget = $target.Replace("'", "''")
$literalLogPath = $logPath.Replace("'", "''")
$command = @"
`$ErrorActionPreference = 'Stop'
Start-Transcript -LiteralPath '$literalLogPath' -Force | Out-Null
`$exitCode = 0
try {
    & '$literalTarget'
} catch {
    Write-Host (`$_ | Format-List * -Force | Out-String) -ForegroundColor Red
    `$exitCode = 1
} finally {
    Stop-Transcript | Out-Null
}
exit `$exitCode
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
Write-Host "Detailed log: $logPath" -ForegroundColor Cyan
$process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded
) -Verb RunAs -Wait -PassThru
exit $process.ExitCode
