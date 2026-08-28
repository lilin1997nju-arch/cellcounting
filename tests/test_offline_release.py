from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_offline_release_builder_pins_git_and_bundles_runtime_assets():
    builder = (ROOT / "scripts" / "build_offline_release.ps1").read_text(encoding="utf-8")
    installer = (ROOT / "deploy" / "offline" / "install_offline.ps1").read_text(encoding="utf-8")
    launcher = (ROOT / "deploy" / "offline" / "Install-CellVision.cmd").read_text(encoding="utf-8")
    elevation = (ROOT / "deploy" / "offline" / "launch_installer.ps1").read_text(encoding="utf-8")
    install_ui = (ROOT / "deploy" / "windows" / "install_ui.ps1").read_text(encoding="utf-8")
    install_ui_bytes = (ROOT / "deploy" / "windows" / "install_ui.ps1").read_bytes()

    assert "status --porcelain" in builder
    assert "git_commit" in builder
    assert "RELEASE_GIT_COMMIT.txt" in builder
    assert "SHA256SUMS.txt" in builder
    assert "prepare_portable_production_runtime.ps1" in builder
    assert 'deployment_mode = "portable-folder"' in builder
    assert 'application_directory = "Application"' in builder
    assert 'workspace_directory = "Workspace"' in builder
    assert 'Join-Path $application "ModelBundle"' in builder
    assert 'Join-Path $releaseRoot "Workspace"' in builder
    assert '"Configure-CellVision-Service.cmd"' in builder
    assert '"Disable-CellVision-Autostart.cmd"' in builder
    assert '"Start-CellVision.cmd"' in builder
    assert '"start_cellvision_platform.ps1"' in builder
    assert '"repair_project_manifests.py"' in builder
    assert '"Recover-CellVision-Projects.cmd"' in builder
    assert '"recover_cellvision_projects.ps1"' in builder
    assert '"recover_cellvision_projects_launcher.ps1"' in builder
    assert '"PROJECT-RECOVERY-README.txt"' in builder
    assert '"Import-CellVision-Workspace.cmd"' in builder
    assert '"import_cellvision_workspace.py"' in builder
    assert '"import_cellvision_workspace.ps1"' in builder
    assert '"import_cellvision_workspace_launcher.ps1"' in builder
    assert '"Open-CellVision.cmd"' in builder
    assert "python-$PythonVersion-amd64.exe" not in builder
    assert '"torch==2.11.0"' in builder
    assert '"torchvision==0.26.0"' in builder
    assert "Reusing prepared portable runtime from" in builder
    assert "Reusing offline wheel assets from" in builder
    assert "Refreshing only missing or changed wheels" in builder
    assert "$fallbackAssetSource" in builder
    assert "runtime_assets_reused_from" in builder
    assert "Get-AssetContract" in builder
    assert "teaching_classifier.pt" in builder
    assert "multiplicity_classifier.pt" in builder
    assert "resnet18-f37072fd.pth" in builder
    assert "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec" in builder
    assert "latest_instance_segmenter.pt" in builder
    assert "latest_temporal_evidence.pt" not in builder

    assert "Get-FileHash" in installer
    assert "Get-Content -LiteralPath $hashPath -Encoding UTF8" in installer
    assert "PACKAGE_VERIFICATION_OK" in installer
    assert '[string]$PackageRoot = ""' in installer
    assert "-Offline" in installer
    assert "-Wheelhouse" in installer
    assert "Move-Item -LiteralPath $InstallRoot" in installer
    assert 'Join-Path $InstallRoot "Start-CellVision.cmd"' in installer
    assert "cellvision.desktop_bridge" in installer
    assert "$shortcut.TargetPath = $rootLauncherPath" in installer
    assert 'GetFolderPath("CommonDesktopDirectory")' in installer
    assert 'install_production_service.ps1' in installer
    assert "InstallAllUsers=1" in installer
    assert "Test-BundledPythonRuntime" in installer
    assert "import encodings, pip, ssl, sys" in installer
    assert '"Include_dev=0"' in installer
    assert '"Include_tcltk=0"' in installer
    assert '"Include_symbols=0"' in installer
    assert '"Include_debug=0"' in installer
    assert "Removing an incomplete bundled Python runtime" in installer
    assert 'Join-Path $env:ProgramData "CellVision\\InstallerLogs"' in installer
    assert "Test-Path -LiteralPath $installParent -PathType Container" in installer
    assert "The installation directory cannot be a drive root" in installer
    assert "requirements-windows-service.txt" in builder
    assert "launch_installer.ps1" in launcher
    assert "-Verb RunAs" in elevation
    assert "-Wait" in elevation
    assert "FolderBrowserDialog" in install_ui
    assert install_ui_bytes.startswith(b"\xef\xbb\xbf")
    assert "INSTALL_UI_PARSE_OK" in install_ui
    assert "INSTALL_UI_SMOKE_OK" in install_ui
    assert '"-InstallRoot", $script:selectedInstallRoot' in install_ui
    assert '"-StartAfterInstall"' in install_ui
    assert "不能直接安装到盘符根目录" in install_ui


def test_single_file_windows_installer_wraps_existing_gui_release():
    builder = (ROOT / "scripts" / "build_single_file_installer.ps1").read_text(encoding="utf-8")

    assert 'ValidateSet("production", "review")' in builder
    assert "System32\\iexpress.exe" in builder
    assert 'Copy-Item -LiteralPath $archivePath -Destination (Join-Path $workingRoot "payload.zip")' in builder
    assert 'AppLaunched=cmd.exe /d /c bootstrap.cmd' in builder
    assert 'Expand-Archive -LiteralPath $payload' in builder
    assert 'Get-ChildItem -LiteralPath $extractRoot -Directory' in builder
    assert '-Filter "__LAUNCHER__" -File -Recurse' not in builder
    assert 'exactly one top-level __LAUNCHER__' in builder
    assert 'SINGLE_FILE_INSTALLER_VALIDATION_OK' in builder


def test_offline_setup_uses_no_index_and_four_model_contract():
    setup = (ROOT / "scripts" / "setup_production.ps1").read_text(encoding="utf-8")
    start = (ROOT / "scripts" / "start_production.ps1").read_text(encoding="utf-8")

    assert '--no-index", "--find-links", $Wheelhouse' in setup
    assert "if ($Offline)" in setup
    for script in (setup, start):
        assert "teaching_classifier.pt" in script
        assert "multiplicity_classifier.pt" in script
        assert "resnet18-f37072fd.pth" in script
        assert "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec" in script
        assert "latest_instance_segmenter.pt" in script
        assert "latest_temporal_evidence.pt" not in script


def test_portable_production_layout_keeps_application_and_workspace_together():
    configure = (ROOT / "deploy" / "portable" / "configure_service.ps1").read_text(encoding="utf-8")
    configure_launcher = (ROOT / "deploy" / "portable" / "configure_service_launcher.ps1").read_text(encoding="utf-8")
    launcher = (ROOT / "deploy" / "portable" / "Open-CellVision.cmd").read_text(encoding="utf-8")
    daily_launcher = (ROOT / "deploy" / "portable" / "start_cellvision_platform.ps1").read_text(encoding="utf-8")
    recovery_launcher = (ROOT / "deploy" / "portable" / "recover_cellvision_projects.ps1").read_text(encoding="utf-8")
    disable_autostart = (ROOT / "deploy" / "portable" / "disable_service_autostart.ps1").read_text(encoding="utf-8")
    zip_installer = (ROOT / "deploy" / "portable" / "install_cellvision_zip.ps1").read_text(encoding="utf-8")
    zip_builder = (ROOT / "scripts" / "build_electron_installer.ps1").read_text(encoding="utf-8")
    prepare = (ROOT / "scripts" / "prepare_portable_production_runtime.ps1").read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup_production.ps1").read_text(encoding="utf-8")

    assert 'Join-Path $deploymentRoot "Application"' in configure
    assert 'Join-Path $deploymentRoot "Workspace"' in configure
    assert "Checking the bundled Python executable" in configure
    assert "Running the full Cell Vision/PyTorch validation" in configure
    assert 'Join-Path $workspaceRoot "Projects"' in configure
    assert "*S-1-5-11:(OI)(CI)M" not in configure
    assert "icacls.exe" not in configure
    assert "Backing up project metadata" in configure
    assert "--recover" in configure
    assert "RepairService" in configure
    assert "Refusing to switch Workspace automatically" in configure
    assert "PORTABLE_SERVICE_CONFIGURATION_VALIDATION_OK" in configure
    assert "pywin32_postinstall" not in configure
    assert 'Join-Path $applicationRoot "Python312\\pythonservice.exe"' in configure
    assert 'Join-Path $env:LOCALAPPDATA "CellVisionInstallerLogs"' in configure_launcher
    assert "Start-Transcript" in configure_launcher
    assert "%~dp0Application\\Python312\\python.exe" in launcher
    assert 'Get-Service -Name $serviceName' in daily_launcher
    assert 'api/project/worker-runtime' in daily_launcher
    assert 'api/catalog/status' in daily_launcher
    assert 'repair_project_manifests.py' in daily_launcher
    assert 'project.json is missing or invalid' in daily_launcher
    assert 'belongs to a different Cell Vision folder' in daily_launcher
    assert 'Start-Service -Name $serviceName' in daily_launcher
    assert 'install_production_service.ps1' not in daily_launcher
    assert 'icacls.exe' not in daily_launcher
    assert '--backup --recover' in recovery_launcher
    assert 'api/catalog/status' in recovery_launcher
    assert '-NoOpen' in recovery_launcher
    assert 'Remove-Item' not in recovery_launcher
    assert "www.nuget.org/api/v2/package/python" in prepare
    assert "cellvision-portable.pth" in prepare
    assert 'Join-Path $runtime "pythonservice.exe"' in prepare
    assert 'Filter "pywintypes*.dll"' in prepare
    assert "UseBasePythonRuntime" in setup
    assert "SkipDependencyInstall" in setup
    assert 'if ($Device -eq "cpu")' in setup
    assert "skipping NVIDIA and WMI video-controller probes" in setup
    assert "OperationTimeoutSec 10" in setup
    assert "runtime loading exceeded 300 seconds" in setup
    assert "Windows security scanning can make the first load slower" in setup
    assert "Checking the four bundled production model files" in setup
    assert 'if (-not $SkipDependencyInstall -and -not $Offline' in setup
    assert "Text.UTF8Encoding($false)" in setup
    assert "Normalized legacy UTF-8 BOM in project manifest" in setup
    assert 'Set-Content -LiteralPath $Manifest -Encoding UTF8' not in setup
    assert '$env:PIP_NO_INDEX = "1"' in configure
    assert "start= demand" in disable_autostart
    assert 'StartMode -ne "Manual"' in disable_autostart
    assert "was not stopped" in disable_autostart
    assert "DISABLE_AUTOSTART_SCRIPT_VALIDATION_OK" in disable_autostart
    assert "CellVisionDesktopProduction" in zip_installer
    assert "vc_redist.x64.exe" in zip_installer
    assert "configure_service.ps1" in zip_installer
    assert "& $configureScript -Device cpu" in zip_installer
    assert '[ValidateSet("auto", "cpu", "cuda")][string]$Device = "cpu"' in configure
    assert "CommonDesktopDirectory" in zip_installer
    assert "CELLVISION_ZIP_INSTALL.json" in zip_installer
    assert "CellVisionDesktopProduction" in zip_installer
    assert "UninstallString" in zip_installer
    assert "Uninstall-CellVision.cmd" in zip_installer
    assert 'ValidateSet("nsis", "zip")' in zip_builder
    assert '"Uninstall-CellVision.cmd"' in zip_builder
    assert '"uninstall_cellvision.ps1"' in zip_builder
    assert '"uninstall_cellvision_cleanup.ps1"' in zip_builder
    assert "CELLVISION_PACKAGE_FILES.txt" in zip_builder
    assert "CELLVISION_DESKTOP_VERSION.txt" in zip_builder
    assert "resnet18-f37072fd.pth" in zip_builder
    assert "Bundled offline ResNet18 weight could not be loaded" in zip_builder
    assert "npm.cmd run pack:dir" in zip_builder
    assert "CellVision-Desktop-$desktopVersion-x64.zip" in zip_builder
