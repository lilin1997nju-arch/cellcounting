from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
PORTABLE = ROOT / "deploy" / "portable"


def test_uninstaller_is_scoped_to_current_instance_and_preserves_workspace():
    uninstaller = (PORTABLE / "uninstall_cellvision.ps1").read_text(encoding="utf-8")
    cleanup = (PORTABLE / "uninstall_cellvision_cleanup.ps1").read_text(
        encoding="utf-8"
    )

    assert 'serviceName = "CellVisionDesktopProduction"' in uninstaller
    assert "Get-ServiceExecutable" in uninstaller
    assert "belongs to another installation and was not changed" in uninstaller
    assert "Workspace will be preserved" in uninstaller
    assert "Remove-MatchingShortcut" in uninstaller
    assert "CellVisionDesktopProduction" in uninstaller
    assert "CELLVISION_PACKAGE_FILES.txt" in uninstaller
    assert "Refusing delayed cleanup at drive root" in cleanup
    assert "Unsafe uninstall manifest entry" in cleanup
    assert "Uninstall target escaped the installation root" in cleanup
    assert 'Join-Path $root "Workspace"' in cleanup
    assert "Remove-Item -LiteralPath $application -Recurse -Force" in cleanup


def test_delayed_cleanup_removes_program_files_but_keeps_workspace(tmp_path):
    deployment = tmp_path / "CellVision Test"
    workspace = deployment / "Workspace"
    application = deployment / "Application"
    workspace.mkdir(parents=True)
    application.mkdir()
    (workspace / "historical-project.txt").write_text("keep", encoding="utf-8")
    (application / "runtime.dll").write_text("remove", encoding="utf-8")
    (deployment / "Cell Vision.exe").write_text("remove", encoding="utf-8")
    log_path = workspace / "cleanup.log"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("Application/runtime.dll\nCell Vision.exe\n", encoding="utf-8")
    cleanup_copy = tmp_path / "cleanup-copy.ps1"
    shutil.copy2(PORTABLE / "uninstall_cellvision_cleanup.ps1", cleanup_copy)

    subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(cleanup_copy),
            "-DeploymentRoot",
            str(deployment),
            "-ManifestPath",
            str(manifest),
            "-LogPath",
            str(log_path),
        ],
        check=True,
        timeout=30,
    )

    assert not application.exists()
    assert not (deployment / "Cell Vision.exe").exists()
    assert (workspace / "historical-project.txt").read_text(encoding="utf-8") == "keep"
    assert "Workspace preserved" in log_path.read_text(encoding="utf-8-sig")
