[CmdletBinding()]
param(
    [string]$Package = "",
    [int]$Port = 8788
)

$ErrorActionPreference = "Stop"
function ConvertTo-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + $Value.Replace('"', '\"') + '"'
}
$installRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$python = Join-Path $installRoot ".venv-review-platform\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Cell Vision Review Platform is not installed correctly: $python"
}
$logs = Join-Path $installRoot "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$arguments = @("-m", "cellvision.review_platform", "--port", [string]$Port)
if (-not [string]::IsNullOrWhiteSpace($Package)) {
    $arguments += @("--package", [IO.Path]::GetFullPath($Package))
}
$quotedArguments = @($arguments | ForEach-Object { ConvertTo-ProcessArgument -Value ([string]$_) })
$process = Start-Process -FilePath $python -ArgumentList $quotedArguments -WorkingDirectory $installRoot `
    -RedirectStandardOutput (Join-Path $logs "platform.stdout.log") `
    -RedirectStandardError (Join-Path $logs "platform.stderr.log") `
    -WindowStyle Hidden -PassThru
Write-Host "Cell Vision Review Platform started (PID $($process.Id))."
