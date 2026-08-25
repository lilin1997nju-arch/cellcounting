from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_electron_client_is_pinned_and_has_no_native_runtime_dependencies():
    package = json.loads((ROOT / "electron" / "package.json").read_text(encoding="utf-8"))
    assert package["devDependencies"]["electron"] == "43.4.1"
    assert package["devDependencies"]["electron-builder"] == "26.15.3"
    assert "dependencies" not in package
    assert package["build"]["nsis"]["perMachine"] is True
    assert package["build"]["nsis"]["deleteAppDataOnUninstall"] is False


def test_nsis_installer_preserves_workspace_and_configures_prerequisites():
    installer = (ROOT / "electron" / "build" / "installer.nsh").read_text(encoding="utf-8")
    assert '!macro customRemoveFiles' in installer
    assert 'RMDir /r "$INSTDIR\\Application"' in installer
    assert 'RMDir /r "$INSTDIR\\Workspace"' not in installer
    assert 'vc_redist.x64.exe' in installer
    assert 'configure_service.ps1' in installer
    assert 'Workspace data was preserved' in installer


def test_electron_build_stages_workspace_tools_but_never_workspace_data():
    builder = (ROOT / "scripts" / "build_electron_installer.ps1").read_text(encoding="utf-8")
    assert 'import_cellvision_workspace.py' in builder
    assert 'repair_project_manifests.py' in builder
    assert 'vc_redist.x64.exe' in builder
    assert 'payloadRoot "Workspace"' not in builder
    assert 'Copying the portable Python/model runtime' in builder
