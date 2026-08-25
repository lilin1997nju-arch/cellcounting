<# Request elevation once, then run the historical Workspace import. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceWorkspace,
    [int]$StartupTimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$importScript = Join-Path $PSScriptRoot "import_cellvision_workspace.ps1"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    & $importScript -SourceWorkspace $SourceWorkspace -StartupTimeoutSeconds $StartupTimeoutSeconds
    exit $LASTEXITCODE
}

$escapedScript = $importScript.Replace("'", "''")
$escapedSource = ([IO.Path]::GetFullPath($SourceWorkspace)).Replace("'", "''")
$command = "& '$escapedScript' -SourceWorkspace '$escapedSource' -StartupTimeoutSeconds $StartupTimeoutSeconds"
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList @(
    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded
) -WorkingDirectory $PSScriptRoot -Wait -PassThru
exit $process.ExitCode
