<# Remove this extracted-folder desktop installation while preserving Workspace. #>
[CmdletBinding()]
param([switch]$Elevated)

$ErrorActionPreference = "Stop"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd("\")
$rootPrefix = $deploymentRoot + "\"
$driveRoot = [IO.Path]::GetPathRoot($deploymentRoot).TrimEnd("\")
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$applicationRoot = Join-Path $deploymentRoot "Application"
$desktopExecutable = Join-Path $deploymentRoot "Cell Vision.exe"
$serviceHost = Join-Path $applicationRoot "Python312\pythonservice.exe"
$manifestPath = Join-Path $deploymentRoot "CELLVISION_PACKAGE_FILES.txt"
$cleanupSource = Join-Path $deploymentRoot "uninstall_cellvision_cleanup.ps1"
$serviceName = "CellVisionDesktopProduction"
$uninstallKey = "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CellVisionDesktopProduction"

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedUninstall {
    $arguments = @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", ('"' + $PSCommandPath + '"'), "-Elevated"
    )
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments `
        -WorkingDirectory $deploymentRoot -Wait -PassThru
    exit $process.ExitCode
}

function Get-ServiceExecutable {
    param([Parameter(Mandatory = $true)]$ServiceInfo)
    $pathName = [string]$ServiceInfo.PathName
    if ($pathName -match '^\s*"([^"]+)"') { return $Matches[1] }
    if ($pathName -match '^\s*([^\s]+)') { return $Matches[1] }
    return ""
}

function Remove-MatchingShortcut {
    param([Parameter(Mandatory = $true)][string]$ShortcutPath)
    if (-not (Test-Path -LiteralPath $ShortcutPath -PathType Leaf)) { return }
    try {
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($ShortcutPath)
        $target = [string]$shortcut.TargetPath
        if (-not [string]::IsNullOrWhiteSpace($target) -and [string]::Equals(
            [IO.Path]::GetFullPath($target),
            [IO.Path]::GetFullPath($desktopExecutable),
            [StringComparison]::OrdinalIgnoreCase
        )) {
            Remove-Item -LiteralPath $ShortcutPath -Force
        }
    } catch {
        Write-Warning "Unable to inspect shortcut $ShortcutPath`: $($_.Exception.Message)"
    }
}

if ([string]::IsNullOrWhiteSpace($deploymentRoot) -or $deploymentRoot -eq $driveRoot) {
    throw "Refusing to uninstall from a drive root: $deploymentRoot"
}
if (-not $deploymentRoot.StartsWith($driveRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "The installation root could not be validated: $deploymentRoot"
}
foreach ($required in @($applicationRoot, $manifestPath, $cleanupSource)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "This folder is not a complete Cell Vision ZIP installation: $required"
    }
}
if (-not (Test-Administrator)) { Invoke-ElevatedUninstall }

New-Item -ItemType Directory -Force -Path (Join-Path $workspaceRoot "Logs") | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $workspaceRoot "Logs\cellvision-uninstall-$stamp.log"
$transcriptStarted = $false
try {
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Uninstalling Cell Vision from: $deploymentRoot" -ForegroundColor Cyan
    Write-Host "Workspace will be preserved: $workspaceRoot" -ForegroundColor Green

    $service = Get-CimInstance Win32_Service -Filter "Name='$serviceName'" -ErrorAction SilentlyContinue
    if ($null -ne $service) {
        $actualServiceHost = Get-ServiceExecutable -ServiceInfo $service
        $sameService = -not [string]::IsNullOrWhiteSpace($actualServiceHost) -and [string]::Equals(
            [IO.Path]::GetFullPath($actualServiceHost),
            [IO.Path]::GetFullPath($serviceHost),
            [StringComparison]::OrdinalIgnoreCase
        )
        if ($sameService) {
            Write-Host "Stopping and removing $serviceName ..." -ForegroundColor Cyan
            $serviceState = Get-Service -Name $serviceName -ErrorAction Stop
            if ($serviceState.Status -ne "Stopped") {
                Stop-Service -Name $serviceName -Force
                (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
            }
            & sc.exe delete $serviceName | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Unable to delete Windows service $serviceName." }
        } else {
            Write-Warning "$serviceName belongs to another installation and was not changed: $actualServiceHost"
        }
    }

    Get-CimInstance Win32_Process -Filter "Name='Cell Vision.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            -not [string]::IsNullOrWhiteSpace([string]$_.ExecutablePath) -and [string]::Equals(
                [IO.Path]::GetFullPath([string]$_.ExecutablePath),
                [IO.Path]::GetFullPath($desktopExecutable),
                [StringComparison]::OrdinalIgnoreCase
            )
        } | ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }

    Remove-MatchingShortcut -ShortcutPath (Join-Path ([Environment]::GetFolderPath("CommonDesktopDirectory")) "Cell Vision.lnk")
    Remove-MatchingShortcut -ShortcutPath (Join-Path ([Environment]::GetFolderPath("CommonPrograms")) "Cell Vision\Cell Vision.lnk")

    if (Test-Path -LiteralPath $uninstallKey) {
        $registeredRoot = [string](Get-ItemProperty -LiteralPath $uninstallKey -Name InstallLocation -ErrorAction SilentlyContinue).InstallLocation
        if ([string]::IsNullOrWhiteSpace($registeredRoot) -or [string]::Equals(
            [IO.Path]::GetFullPath($registeredRoot).TrimEnd("\"),
            $deploymentRoot,
            [StringComparison]::OrdinalIgnoreCase
        )) {
            Remove-Item -LiteralPath $uninstallKey -Recurse -Force
        }
    }

    $preservedRecord = Join-Path $workspaceRoot "CELLVISION_UNINSTALLED.txt"
    [IO.File]::WriteAllText(
        $preservedRecord,
        ("Cell Vision application uninstalled at {0}. Workspace project data was preserved.{1}" -f `
            (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), [Environment]::NewLine),
        (New-Object Text.UTF8Encoding($false))
    )

    $cleanupId = [guid]::NewGuid().ToString("N")
    $temporaryCleanup = Join-Path $env:TEMP "CellVision-Uninstall-$cleanupId.ps1"
    $temporaryManifest = Join-Path $env:TEMP "CellVision-Uninstall-$cleanupId-files.txt"
    Copy-Item -LiteralPath $cleanupSource -Destination $temporaryCleanup -Force
    Copy-Item -LiteralPath $manifestPath -Destination $temporaryManifest -Force
    $literalCleanup = $temporaryCleanup.Replace("'", "''")
    $literalRoot = $deploymentRoot.Replace("'", "''")
    $literalManifest = $temporaryManifest.Replace("'", "''")
    $literalLog = $logPath.Replace("'", "''")
    $cleanupCommand = (
        "& '$literalCleanup' -DeploymentRoot '$literalRoot' " +
        "-ManifestPath '$literalManifest' -LogPath '$literalLog'"
    )
    $encodedCleanup = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cleanupCommand))
    Write-Host "Program-file cleanup has been scheduled. Workspace will remain in place." -ForegroundColor Green
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCleanup
    ) -WindowStyle Hidden | Out-Null
} catch {
    Write-Host ($_ | Format-List * -Force | Out-String) -ForegroundColor Red
    exit 1
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
exit 0
