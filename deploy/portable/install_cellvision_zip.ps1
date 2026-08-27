<# Configure an already-extracted Cell Vision desktop ZIP in its final directory. #>
[CmdletBinding()]
param(
    [switch]$Elevated,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$desktopExecutable = Join-Path $deploymentRoot "Cell Vision.exe"
$configureScript = Join-Path $deploymentRoot "configure_service.ps1"
$redistributable = Join-Path $deploymentRoot "vc_redist.x64.exe"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedInstall {
    $arguments = @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ('"' + $PSCommandPath + '"'), "-Elevated"
    )
    if ($NoLaunch) { $arguments += "-NoLaunch" }
    Write-Host "Administrator permission is required to configure the machine service." -ForegroundColor Yellow
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments `
        -WorkingDirectory $deploymentRoot -Wait -PassThru
    exit $process.ExitCode
}

function New-CellVisionShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$ShortcutPath
    )
    $parent = Split-Path -Parent $ShortcutPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $desktopExecutable
    $shortcut.WorkingDirectory = $deploymentRoot
    $shortcut.IconLocation = "$desktopExecutable,0"
    $shortcut.Description = "Cell Vision"
    $shortcut.Save()
}

foreach ($required in @(
    $desktopExecutable,
    $configureScript,
    $redistributable,
    (Join-Path $applicationRoot "Python312\python.exe"),
    (Join-Path $applicationRoot "ModelBundle")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw (
            "The Cell Vision ZIP is not fully extracted. Missing: $required`n" +
            "Use Extract All first, then run Install-CellVision.cmd from the extracted folder."
        )
    }
}

if (-not (Test-Administrator)) { Invoke-ElevatedInstall }

New-Item -ItemType Directory -Force -Path $workspaceRoot | Out-Null
$logRoot = Join-Path $workspaceRoot "Logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot ("cellvision-zip-install-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$transcriptStarted = $false
$exitCode = 0
try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Installing Cell Vision from extracted folder: $deploymentRoot" -ForegroundColor Cyan

    Write-Host "Granting local users access to the preserved Workspace ..." -ForegroundColor Cyan
    & icacls.exe $workspaceRoot /inheritance:e /grant '*S-1-5-32-545:(OI)(CI)M' /C /Q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Unable to grant local-user access to Workspace." }

    Write-Host "Checking the Microsoft Visual C++ x64 runtime ..." -ForegroundColor Cyan
    $signature = Get-AuthenticodeSignature -LiteralPath $redistributable
    if ($signature.Status -ne "Valid" -or [string]$signature.SignerCertificate.Subject -notmatch "Microsoft") {
        throw "The bundled Microsoft Visual C++ runtime signature is invalid."
    }
    $runtimeProcess = Start-Process -FilePath $redistributable `
        -ArgumentList @("/install", "/quiet", "/norestart") -Wait -PassThru
    if ($runtimeProcess.ExitCode -notin @(0, 1638, 3010)) {
        throw "Microsoft Visual C++ runtime installation failed with exit code $($runtimeProcess.ExitCode)."
    }

    Write-Host "Configuring the desktop service and selecting an available local port ..." -ForegroundColor Cyan
    & $configureScript
    if ($LASTEXITCODE -ne 0) { throw "Cell Vision service configuration failed." }

    Write-Host "Creating desktop and Start Menu shortcuts ..." -ForegroundColor Cyan
    $commonDesktop = [Environment]::GetFolderPath("CommonDesktopDirectory")
    $commonPrograms = [Environment]::GetFolderPath("CommonPrograms")
    New-CellVisionShortcut -ShortcutPath (Join-Path $commonDesktop "Cell Vision.lnk")
    New-CellVisionShortcut -ShortcutPath (Join-Path $commonPrograms "Cell Vision\Cell Vision.lnk")

    $installationRecord = [ordered]@{
        format = "cellvision-extracted-zip"
        version = 1
        installed_at = (Get-Date).ToUniversalTime().ToString("o")
        deployment_root = $deploymentRoot
        workspace = $workspaceRoot
        service = "CellVisionDesktopProduction"
    }
    $installationRecord | ConvertTo-Json -Depth 3 | Set-Content `
        -LiteralPath (Join-Path $deploymentRoot "CELLVISION_ZIP_INSTALL.json") -Encoding UTF8

    Write-Host "Cell Vision ZIP installation completed successfully." -ForegroundColor Green
    if (-not $NoLaunch) {
        Start-Process -FilePath $desktopExecutable -WorkingDirectory $deploymentRoot
    }
} catch {
    $exitCode = 1
    Write-Host ($_ | Format-List * -Force | Out-String) -ForegroundColor Red
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
Write-Host "Detailed log: $logPath"
exit $exitCode
