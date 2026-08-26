from __future__ import annotations

from cellvision.project_server import _production_instance_id
from cellvision.windows_service import configured_service_name


def test_production_instance_prefers_environment(monkeypatch, tmp_path):
    (tmp_path / ".cellvision-instance-id").write_text("workspace-id\n", encoding="utf-8")
    monkeypatch.setenv("CELLVISION_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("CELLVISION_INSTANCE_ID", "environment-id")

    assert _production_instance_id() == "environment-id"


def test_production_instance_falls_back_to_workspace_file(monkeypatch, tmp_path):
    (tmp_path / ".cellvision-instance-id").write_text("workspace-id\n", encoding="utf-8")
    monkeypatch.delenv("CELLVISION_INSTANCE_ID", raising=False)
    monkeypatch.setenv("CELLVISION_ARTIFACT_ROOT", str(tmp_path))

    assert _production_instance_id() == "workspace-id"


def test_windows_service_name_can_be_scoped_to_the_desktop_install(monkeypatch, tmp_path):
    monkeypatch.delenv("CELLVISION_SERVICE_NAME", raising=False)
    (tmp_path / ".env.production").write_text(
        "CELLVISION_SERVICE_NAME=CellVisionDesktopProduction\n",
        encoding="utf-8",
    )

    assert configured_service_name(tmp_path) == "CellVisionDesktopProduction"
