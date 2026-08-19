[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("production", "review")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function ConvertTo-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Get-ExistingParent {
    param([Parameter(Mandatory = $true)][string]$Path)
    $candidate = [IO.Path]::GetFullPath($Path)
    while (-not (Test-Path -LiteralPath $candidate -PathType Container)) {
        $parent = [IO.Path]::GetDirectoryName($candidate)
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $candidate) { break }
        $candidate = $parent
    }
    return $candidate
}

$packageRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$isProduction = $Mode -eq "production"
$productName = if ($isProduction) { "Cell Vision 生产平台" } else { "Cell Vision 审核平台" }
$installerName = if ($isProduction) { "install_offline.ps1" } else { "install_review_platform.ps1" }
$installerPath = Join-Path $packageRoot $installerName
$defaultInstallRoot = if ($isProduction) {
    Join-Path $env:LOCALAPPDATA "CellVision"
} else {
    Join-Path $env:LOCALAPPDATA "CellVisionReviewPlatform"
}

if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
    [Windows.Forms.MessageBox]::Show(
        "安装文件不完整。请先对发布 ZIP 执行'全部解压'，然后从解压后的文件夹运行安装程序。`n`n缺少：$installerName",
        "$productName 安装失败",
        [Windows.Forms.MessageBoxButtons]::OK,
        [Windows.Forms.MessageBoxIcon]::Error
    ) | Out-Null
    exit 1
}

$form = New-Object Windows.Forms.Form
$form.Text = "$productName 安装向导"
$form.StartPosition = "CenterScreen"
$form.ClientSize = New-Object Drawing.Size(720, 470)
$form.MinimumSize = New-Object Drawing.Size(736, 509)
$form.Font = New-Object Drawing.Font("Microsoft YaHei UI", 9)
$form.BackColor = [Drawing.Color]::FromArgb(246, 247, 249)
$form.TopMost = $true

$title = New-Object Windows.Forms.Label
$title.Text = "安装 $productName"
$title.Font = New-Object Drawing.Font("Microsoft YaHei UI", 18, [Drawing.FontStyle]::Bold)
$title.ForeColor = [Drawing.Color]::FromArgb(28, 52, 58)
$title.Location = New-Object Drawing.Point(28, 22)
$title.AutoSize = $true
$form.Controls.Add($title)

$description = New-Object Windows.Forms.Label
$description.Text = if ($isProduction) {
    "请选择应用安装目录。模型、运行环境和程序文件会安装到该目录；任务数据仍保存在独立的数据目录。"
} else {
    "请选择审核工具安装目录。.cvreview 审核数据可以存放在电脑的任意位置，不会复制到安装目录。"
}
$description.ForeColor = [Drawing.Color]::FromArgb(88, 102, 108)
$description.Location = New-Object Drawing.Point(31, 67)
$description.Size = New-Object Drawing.Size(650, 42)
$form.Controls.Add($description)

$pathLabel = New-Object Windows.Forms.Label
$pathLabel.Text = "安装目录"
$pathLabel.Location = New-Object Drawing.Point(31, 119)
$pathLabel.AutoSize = $true
$form.Controls.Add($pathLabel)

$pathText = New-Object Windows.Forms.TextBox
$pathText.Text = $defaultInstallRoot
$pathText.Location = New-Object Drawing.Point(31, 143)
$pathText.Size = New-Object Drawing.Size(550, 27)
$form.Controls.Add($pathText)

$browseButton = New-Object Windows.Forms.Button
$browseButton.Text = "浏览…"
$browseButton.Location = New-Object Drawing.Point(593, 141)
$browseButton.Size = New-Object Drawing.Size(92, 30)
$form.Controls.Add($browseButton)

$launchAfterInstall = New-Object Windows.Forms.CheckBox
$launchAfterInstall.Text = "安装完成后启动 $productName"
$launchAfterInstall.Checked = $true
$launchAfterInstall.Location = New-Object Drawing.Point(31, 184)
$launchAfterInstall.Size = New-Object Drawing.Size(360, 26)
$form.Controls.Add($launchAfterInstall)

$progress = New-Object Windows.Forms.ProgressBar
$progress.Location = New-Object Drawing.Point(31, 221)
$progress.Size = New-Object Drawing.Size(654, 18)
$progress.Style = [Windows.Forms.ProgressBarStyle]::Blocks
$form.Controls.Add($progress)

$statusLabel = New-Object Windows.Forms.Label
$statusLabel.Text = "准备安装"
$statusLabel.Location = New-Object Drawing.Point(31, 250)
$statusLabel.Size = New-Object Drawing.Size(654, 24)
$statusLabel.ForeColor = [Drawing.Color]::FromArgb(52, 83, 90)
$form.Controls.Add($statusLabel)

$logBox = New-Object Windows.Forms.TextBox
$logBox.Location = New-Object Drawing.Point(31, 278)
$logBox.Size = New-Object Drawing.Size(654, 116)
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Vertical"
$logBox.BackColor = [Drawing.Color]::White
$logBox.Text = "安装开始后，这里会显示最近的执行信息。"
$form.Controls.Add($logBox)

$installButton = New-Object Windows.Forms.Button
$installButton.Text = "开始安装"
$installButton.Location = New-Object Drawing.Point(477, 410)
$installButton.Size = New-Object Drawing.Size(100, 36)
$installButton.BackColor = [Drawing.Color]::FromArgb(16, 119, 125)
$installButton.ForeColor = [Drawing.Color]::White
$installButton.FlatStyle = "Flat"
$form.Controls.Add($installButton)

$closeButton = New-Object Windows.Forms.Button
$closeButton.Text = "取消"
$closeButton.Location = New-Object Drawing.Point(585, 410)
$closeButton.Size = New-Object Drawing.Size(100, 36)
$form.Controls.Add($closeButton)

$timer = New-Object Windows.Forms.Timer
$timer.Interval = 500
$script:installing = $false
$script:installProcess = $null
$script:exitCode = 0
$script:stdoutPath = ""
$script:stderrPath = ""
$script:selectedInstallRoot = ""

function Update-InstallLog {
    $lines = @()
    foreach ($logPath in @($script:stdoutPath, $script:stderrPath)) {
        if (-not [string]::IsNullOrWhiteSpace($logPath) -and (Test-Path -LiteralPath $logPath -PathType Leaf)) {
            $lines += @(Get-Content -LiteralPath $logPath -Tail 45 -ErrorAction SilentlyContinue)
        }
    }
    if ($lines.Count -gt 0) {
        $logBox.Lines = @($lines | Select-Object -Last 45)
        $logBox.SelectionStart = $logBox.TextLength
        $logBox.ScrollToCaret()
    }
}

$browseButton.Add_Click({
    $dialog = New-Object Windows.Forms.FolderBrowserDialog
    $dialog.Description = "选择 $productName 的安装目录"
    $dialog.ShowNewFolderButton = $true
    try {
        $dialog.SelectedPath = Get-ExistingParent -Path $pathText.Text.Trim()
        if ($dialog.ShowDialog($form) -eq [Windows.Forms.DialogResult]::OK) {
            $pathText.Text = $dialog.SelectedPath
        }
    } finally {
        $dialog.Dispose()
    }
})

$installButton.Add_Click({
    try {
        if ([string]::IsNullOrWhiteSpace($pathText.Text)) {
            throw "请选择安装目录。"
        }
        $script:selectedInstallRoot = [IO.Path]::GetFullPath(
            [Environment]::ExpandEnvironmentVariables($pathText.Text.Trim())
        )
        if (Test-Path -LiteralPath $script:selectedInstallRoot) {
            $answer = [Windows.Forms.MessageBox]::Show(
                "安装目录已经存在。继续安装将更新该目录中的程序文件。是否继续？`n`n$script:selectedInstallRoot",
                "确认安装目录",
                [Windows.Forms.MessageBoxButtons]::YesNo,
                [Windows.Forms.MessageBoxIcon]::Question
            )
            if ($answer -ne [Windows.Forms.DialogResult]::Yes) { return }
        }

        $logRoot = Join-Path $env:LOCALAPPDATA "CellVisionInstallerLogs"
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $script:stdoutPath = Join-Path $logRoot "$Mode-$stamp.stdout.log"
        $script:stderrPath = Join-Path $logRoot "$Mode-$stamp.stderr.log"

        $arguments = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", $installerPath,
            "-InstallRoot", $script:selectedInstallRoot
        )
        if ($isProduction -and $launchAfterInstall.Checked) {
            $arguments += "-StartAfterInstall"
        }
        $quotedArguments = @($arguments | ForEach-Object { ConvertTo-ProcessArgument -Value ([string]$_) })
        $script:installProcess = Start-Process -FilePath "powershell.exe" `
            -ArgumentList $quotedArguments `
            -WorkingDirectory $packageRoot `
            -RedirectStandardOutput $script:stdoutPath `
            -RedirectStandardError $script:stderrPath `
            -WindowStyle Hidden `
            -PassThru

        $script:installing = $true
        $pathText.Enabled = $false
        $browseButton.Enabled = $false
        $launchAfterInstall.Enabled = $false
        $installButton.Enabled = $false
        $closeButton.Enabled = $false
        $progress.Style = [Windows.Forms.ProgressBarStyle]::Marquee
        $statusLabel.Text = "正在安装，请勿关闭窗口…"
        $logBox.Text = "正在启动安装程序…"
        $timer.Start()
    } catch {
        $script:exitCode = 1
        [Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            "$productName 安装失败",
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$timer.Add_Tick({
    if (-not $script:installing -or $null -eq $script:installProcess) { return }
    Update-InstallLog
    $script:installProcess.Refresh()
    if (-not $script:installProcess.HasExited) { return }

    $timer.Stop()
    $script:installing = $false
    $progress.Style = [Windows.Forms.ProgressBarStyle]::Blocks
    $closeButton.Enabled = $true
    $closeButton.Text = "关闭"
    Update-InstallLog
    if ($script:installProcess.ExitCode -eq 0) {
        $script:exitCode = 0
        $progress.Value = 100
        $statusLabel.Text = "安装完成：$script:selectedInstallRoot"
        if (-not $isProduction -and $launchAfterInstall.Checked) {
            $reviewLauncher = Join-Path $script:selectedInstallRoot "Open-CellVision-Review.cmd"
            if (Test-Path -LiteralPath $reviewLauncher -PathType Leaf) {
                Start-Process -FilePath $reviewLauncher -WorkingDirectory $script:selectedInstallRoot -WindowStyle Hidden
            }
        }
        [Windows.Forms.MessageBox]::Show(
            "$productName 已安装完成。`n`n安装目录：$script:selectedInstallRoot",
            "安装完成",
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    } else {
        $script:exitCode = 1
        $statusLabel.Text = "安装失败，请查看下方信息"
        [Windows.Forms.MessageBox]::Show(
            "安装程序退出代码：$($script:installProcess.ExitCode)`n`n日志：`n$script:stdoutPath`n$script:stderrPath",
            "$productName 安装失败",
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
})

$closeButton.Add_Click({ $form.Close() })
$form.Add_FormClosing({
    param($sender, $eventArgs)
    if ($script:installing) {
        $eventArgs.Cancel = $true
        [Windows.Forms.MessageBox]::Show(
            "安装仍在进行，请等待安装完成。",
            $productName,
            [Windows.Forms.MessageBoxButtons]::OK,
            [Windows.Forms.MessageBoxIcon]::Information
        ) | Out-Null
    }
})
$form.Add_Shown({
    $form.Activate()
    $form.BringToFront()
    $pathText.Focus()
})

[Windows.Forms.Application]::EnableVisualStyles()
[Windows.Forms.Application]::Run($form)
$timer.Dispose()
$form.Dispose()
exit $script:exitCode
