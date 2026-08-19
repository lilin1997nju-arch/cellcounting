from pathlib import Path

from cellvision.windows_service import (
    load_production_environment,
    production_command,
    service_process_environment,
)


ROOT = Path(__file__).parents[1]


def test_service_reads_shared_production_environment_and_builds_worker_command(tmp_path: Path):
    (tmp_path / ".env.production").write_text(
        "CELLVISION_MANIFEST=D:/CellVisionState/projects/active/project.json\n"
        "CELLVISION_PORT=8777\nCELLVISION_WORKER_DEVICE=cpu\n",
        encoding="utf-8",
    )
    python = tmp_path / ".venv-production" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()

    values = load_production_environment(tmp_path)
    command = production_command(tmp_path, values)

    assert command[0] == str(python)
    assert command[1:4] == ["-m", "cellvision", "review-project"]
    assert "D:/CellVisionState/projects/active/project.json" in command
    assert command[-1] == "cpu"
    environment = service_process_environment(tmp_path, values)
    assert environment["CELLVISION_MACHINE_SERVICE"] == "1"


def test_machine_service_installer_sets_delayed_start_and_recovery():
    installer = (ROOT / "scripts" / "install_production_service.ps1").read_text(encoding="utf-8")
    manager = (ROOT / "scripts" / "manage_production_service.py").read_text(encoding="utf-8")

    assert "--startup delayed install" in installer
    assert "sc.exe failure" in installer
    assert 'Get-Service -Name $serviceName' in installer
    assert "win32serviceutil.HandleCommandLine" in manager
