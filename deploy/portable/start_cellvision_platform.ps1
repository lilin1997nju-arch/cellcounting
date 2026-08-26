<# Daily launcher: ensure the installed service and worker are healthy, then open Cell Vision. #>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [ValidateRange(15, 600)][int]$StartupTimeoutSeconds = 120,
    [switch]$Elevated,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$serviceName = "CellVisionDesktopProduction"
$deploymentRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$applicationRoot = Join-Path $deploymentRoot "Application"
$workspaceRoot = Join-Path $deploymentRoot "Workspace"
$logRoot = Join-Path $workspaceRoot "Logs"
$python = Join-Path $applicationRoot "Python312\python.exe"
$expectedServiceHost = Join-Path $applicationRoot "Python312\pythonservice.exe"
$environmentPath = Join-Path $applicationRoot ".env.production"
$instanceIdPath = Join-Path $workspaceRoot ".cellvision-instance-id"
$recoveryScript = Join-Path $deploymentRoot "repair_project_manifests.py"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot "cellvision-daily-launch.log"

function Write-LaunchStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ("[{0}] {1}" -f $timestamp, $Message) -ForegroundColor $Color
    try {
        Add-Content -LiteralPath $logPath -Value ("[{0}] {1}" -f $timestamp, $Message) -Encoding UTF8
    } catch {}
}

trap {
    Write-LaunchStatus ("ERROR: " + $_.Exception.Message) Red
    exit 1
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-PersistedPort {
    if (Test-Path -LiteralPath $environmentPath -PathType Leaf) {
        $line = Get-Content -LiteralPath $environmentPath | Where-Object {
            $_ -match '^CELLVISION_PORT=(\d+)\s*$'
        } | Select-Object -First 1
        if ($line -match '^CELLVISION_PORT=(\d+)\s*$') { return [int]$Matches[1] }
    }
    return 8777
}

function Get-ConfiguredPort {
    if ($Port -gt 0) { return $Port }
    return Get-PersistedPort
}

function Set-ConfiguredPort {
    param([Parameter(Mandatory = $true)][int]$Value)
    $lines = @(Get-Content -LiteralPath $environmentPath -ErrorAction Stop)
    $updated = $false
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match '^CELLVISION_PORT=') {
            $lines[$index] = "CELLVISION_PORT=$Value"
            $updated = $true
            break
        }
    }
    if (-not $updated) { $lines += "CELLVISION_PORT=$Value" }
    [IO.File]::WriteAllText(
        $environmentPath,
        (($lines -join [Environment]::NewLine) + [Environment]::NewLine),
        (New-Object Text.UTF8Encoding($false))
    )
}

function Ensure-InstanceId {
    New-Item -ItemType Directory -Force -Path $workspaceRoot | Out-Null
    if (-not (Test-Path -LiteralPath $instanceIdPath -PathType Leaf)) {
        [IO.File]::WriteAllText(
            $instanceIdPath,
            ([guid]::NewGuid().ToString("D") + [Environment]::NewLine),
            (New-Object Text.UTF8Encoding($false))
        )
    }
    $value = (Get-Content -LiteralPath $instanceIdPath -Raw -Encoding UTF8).Trim()
    if ([string]::IsNullOrWhiteSpace($value)) { throw "Cell Vision instance ID is empty: $instanceIdPath" }
    return $value
}

function Test-TcpPortAvailable {
    param([Parameter(Mandatory = $true)][int]$Candidate)
    $listener = $null
    try {
        $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Candidate)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) { try { $listener.Stop() } catch {} }
    }
}

function Find-AvailablePort {
    param([Parameter(Mandatory = $true)][int]$Preferred)
    for ($offset = 1; $offset -le 1000; $offset++) {
        $candidate = $Preferred + $offset
        if ($candidate -gt 65535) { break }
        if (Test-TcpPortAvailable -Candidate $candidate) { return $candidate }
    }
    throw "No free local TCP port was found after port $Preferred."
}

function Test-InstanceReady {
    param(
        [Parameter(Mandatory = $true)][int]$ReadyPort,
        [Parameter(Mandatory = $true)][string]$ExpectedInstanceId
    )
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$ReadyPort/api/ready" -TimeoutSec 2
        return [string]$response.status -eq "ready" -and `
            [string]$response.instance_id -eq $ExpectedInstanceId
    } catch {
        return $false
    }
}

function Get-ServiceExecutable {
    $serviceInfo = Get-CimInstance Win32_Service -Filter "Name='$serviceName'" -ErrorAction Stop
    $pathName = [string]$serviceInfo.PathName
    $executable = ""
    if ($pathName -match '^\s*"([^"]+)"') { $executable = $Matches[1] }
    elseif ($pathName -match '^\s*([^\s]+)') { $executable = $Matches[1] }
    if ([string]::IsNullOrWhiteSpace($executable)) {
        throw "The installed service executable path is empty."
    }
    return [IO.Path]::GetFullPath($executable)
}

function Invoke-ElevatedLauncher {
    $arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" +
        " -Port $Port -StartupTimeoutSeconds $StartupTimeoutSeconds -Elevated"
    if ($NoOpen) { $arguments += " -NoOpen" }
    Write-LaunchStatus "Administrator permission is needed once to start the Windows service." Yellow
    $process = Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList $arguments `
        -WorkingDirectory $deploymentRoot -Wait -PassThru
    exit $process.ExitCode
}

function Test-WorkerRuntime {
    param([Parameter(Mandatory = $true)][string]$BaseUrl)
    try {
        $runtime = Invoke-RestMethod -Uri ($BaseUrl + "api/project/worker-runtime") -TimeoutSec 2
    } catch {
        return $false
    }
    $status = [string]$runtime.status
    if ($status -in @("", "offline", "unknown", "stopped")) { return $false }
    $workerPid = 0
    try { $workerPid = [int]$runtime.pid } catch { return $false }
    if ($workerPid -le 0 -or $null -eq (Get-Process -Id $workerPid -ErrorAction SilentlyContinue)) {
        return $false
    }
    if ($status -in @("idle", "starting")) {
        try {
            $updated = [DateTimeOffset]::Parse([string]$runtime.updated_at)
            if (([DateTimeOffset]::Now - $updated).TotalSeconds -gt 30) { return $false }
        } catch {
            return $false
        }
    }
    return $true
}

function Show-ProjectManifestAudit {
    $projectsRoot = Join-Path $workspaceRoot "Projects"
    if (-not (Test-Path -LiteralPath $projectsRoot -PathType Container)) { return }
    $validCount = 0
    $incomplete = New-Object System.Collections.Generic.List[string]
    foreach ($directory in Get-ChildItem -LiteralPath $projectsRoot -Directory -ErrorAction SilentlyContinue) {
        if ($directory.Name -eq ".metadata-backups") { continue }
        $manifest = Join-Path $directory.FullName "project.json"
        if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
            if ($directory.Name -ne "active" -and $null -ne (
                Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction SilentlyContinue |
                    Select-Object -First 1
            )) {
                $incomplete.Add($directory.Name)
            }
            continue
        }
        try {
            $null = Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
            $validCount += 1
        } catch {
            $incomplete.Add($directory.Name + " (invalid project.json)")
        }
    }
    Write-LaunchStatus "Project manifest audit: $validCount valid project folder(s)." DarkGray
    if ($incomplete.Count -gt 0) {
        Write-LaunchStatus (
            "Project data folder(s) are not visible on the homepage because project.json is missing or invalid: " +
            ($incomplete -join ", ")
        ) Yellow
    }
}

Write-LaunchStatus "Starting the Cell Vision daily launcher." Cyan
foreach ($required in @($python, $expectedServiceHost, $environmentPath, $recoveryScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Cell Vision is not configured in this folder. Missing: $required"
    }
}

$service = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
if ($null -eq $service) {
    throw "The $serviceName service is not installed. Run Configure-CellVision-Service.cmd once."
}

$actualServiceHost = Get-ServiceExecutable
if (-not [string]::Equals(
    $actualServiceHost,
    [IO.Path]::GetFullPath($expectedServiceHost),
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw (
        "The Windows service belongs to a different Cell Vision folder.`n" +
        "Installed: $actualServiceHost`n" +
        "This entry: $expectedServiceHost`n" +
        "Use the launcher from the installed folder or reconfigure the intended copy once."
    )
}

$instanceId = Ensure-InstanceId
$persistedPort = Get-PersistedPort
$readyPort = Get-ConfiguredPort
$portConfigurationChanged = $false
if ($Port -gt 0 -and $readyPort -ne $persistedPort) {
    if (-not (Test-Administrator)) { Invoke-ElevatedLauncher }
    Set-ConfiguredPort -Value $readyPort
    $portConfigurationChanged = $true
    Write-LaunchStatus "Updated this installation to use port $readyPort." Yellow
}

$currentInstanceReady = Test-InstanceReady -ReadyPort $readyPort -ExpectedInstanceId $instanceId
if (-not $currentInstanceReady -and -not (Test-TcpPortAvailable -Candidate $readyPort)) {
    $occupiedPort = $readyPort
    $readyPort = Find-AvailablePort -Preferred $occupiedPort
    $Port = $readyPort
    if (-not (Test-Administrator)) { Invoke-ElevatedLauncher }
    Set-ConfiguredPort -Value $readyPort
    $portConfigurationChanged = $true
    Write-LaunchStatus (
        "Port $occupiedPort belongs to another process or Cell Vision installation; " +
        "this installation was moved to port $readyPort."
    ) Yellow
}

if ($portConfigurationChanged -and $service.Status -ne "Stopped") {
    Write-LaunchStatus "Restarting $serviceName to apply the new port ..." Cyan
    Stop-Service -Name $serviceName -Force
    (Get-Service -Name $serviceName).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(30))
    $service = Get-Service -Name $serviceName -ErrorAction Stop
}

if ($service.Status -eq "Stopped") {
    if (-not (Test-Administrator)) { Invoke-ElevatedLauncher }
    Write-LaunchStatus "Starting $serviceName ..." Cyan
    Start-Service -Name $serviceName
} elseif ($service.Status -eq "StartPending") {
    Write-LaunchStatus "$serviceName is already starting ..." DarkYellow
} elseif ($service.Status -ne "Running") {
    throw "$serviceName is in unsupported state: $($service.Status)"
} else {
    Write-LaunchStatus "$serviceName is already running; no reinstall or restart is needed." Green
}

$baseUrl = "http://127.0.0.1:$readyPort/"
$timer = [Diagnostics.Stopwatch]::StartNew()
$lastNoticeSecond = -1
$platformReady = $false
while ($timer.Elapsed.TotalSeconds -lt $StartupTimeoutSeconds) {
    $service = Get-Service -Name $serviceName -ErrorAction Stop
    if ($service.Status -eq "Stopped") {
        throw "$serviceName stopped before the platform became ready."
    }
    $apiReady = $false
    try {
        $response = Invoke-RestMethod -Uri ($baseUrl + "api/ready") -TimeoutSec 2
        $apiReady = [string]$response.status -eq "ready" -and `
            [string]$response.instance_id -eq $instanceId
    } catch {}
    if ($apiReady -and (Test-WorkerRuntime -BaseUrl $baseUrl)) {
        $platformReady = $true
        break
    }
    $elapsedSecond = [int]$timer.Elapsed.TotalSeconds
    if ($elapsedSecond -eq 0 -or ($elapsedSecond - $lastNoticeSecond) -ge 5) {
        Write-LaunchStatus "Waiting for the API and compute worker ($elapsedSecond/$StartupTimeoutSeconds seconds) ..." DarkYellow
        $lastNoticeSecond = $elapsedSecond
    }
    Start-Sleep -Seconds 1
}
if (-not $platformReady) {
    throw (
        "The service is running, but the API or compute worker did not become healthy within " +
        "$StartupTimeoutSeconds seconds. Log: $logPath"
    )
}

Write-LaunchStatus "Cell Vision API and compute worker are ready on port $readyPort." Green
try {
    $recoveryJson = & $python $recoveryScript --projects-root (Join-Path $workspaceRoot "Projects") --recover
    if ($LASTEXITCODE -ne 0) { throw "recovery process exited with code $LASTEXITCODE" }
    $recovery = $recoveryJson | ConvertFrom-Json
    $restored = @($recovery.recovery.restored_from_backup).Count
    $reconstructed = @($recovery.recovery.reconstructed_from_catalog).Count
    if ($restored -gt 0 -or $reconstructed -gt 0) {
        Write-LaunchStatus "Recovered project metadata: $restored from backup, $reconstructed from catalog." Green
    }
} catch {
    Write-LaunchStatus "Project metadata recovery was skipped: $($_.Exception.Message)" Yellow
}
try {
    $null = Invoke-RestMethod -Uri ($baseUrl + "api/catalog/status") -TimeoutSec 30
    Write-LaunchStatus "Project catalog refresh completed." DarkGray
} catch {
    Write-LaunchStatus "Project catalog refresh was skipped: $($_.Exception.Message)" Yellow
}
Show-ProjectManifestAudit
if (-not $NoOpen) {
    Write-LaunchStatus "Opening Cell Vision ..." Cyan
    Start-Process -FilePath $python -ArgumentList @(
        "-m", "cellvision.desktop_bridge", "--production-url", $baseUrl
    ) -WorkingDirectory $applicationRoot -WindowStyle Hidden
}
Write-LaunchStatus "Daily startup completed. Log: $logPath" Green
exit 0
