"""Windows Service host for the machine-wide Cell Vision production engine."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:  # Imported only by the dedicated Windows service runtime.
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError:  # Keep development and non-Windows imports functional.
    servicemanager = None
    win32event = None
    win32service = None
    win32serviceutil = None


SERVICE_NAME = "CellVisionProduction"
SERVICE_DISPLAY_NAME = "Cell Vision Production Engine"


def installation_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_production_environment(root: Path) -> dict[str, str]:
    env_path = root / ".env.production"
    if not env_path.is_file():
        raise RuntimeError(f"Production environment file is missing: {env_path}")
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def production_command(root: Path, values: dict[str, str]) -> list[str]:
    configured_python = values.get("CELLVISION_PYTHON", "").strip()
    python = Path(configured_python) if configured_python else (
        root / ".venv-production" / "Scripts" / "python.exe"
    )
    manifest = values.get("CELLVISION_MANIFEST", "")
    if not python.is_file():
        raise RuntimeError(f"Production Python is missing: {python}")
    if not manifest:
        raise RuntimeError("CELLVISION_MANIFEST is missing from .env.production")
    return [
        str(python),
        "-m",
        "cellvision",
        "review-project",
        "--manifest",
        manifest,
        "--host",
        values.get("CELLVISION_HOST", "127.0.0.1"),
        "--port",
        values.get("CELLVISION_PORT", "8777"),
        "--worker-device",
        values.get("CELLVISION_WORKER_DEVICE", "auto"),
    ]


def service_process_environment(root: Path, values: dict[str, str]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(values)
    environment["CELLVISION_PRODUCTION"] = "1"
    environment["CELLVISION_MACHINE_SERVICE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    cache_root = Path(values.get("CELLVISION_LOG_ROOT", str(root / "logs"))).parent / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment["TEMP"] = str(cache_root)
    environment["TMP"] = str(cache_root)
    environment["TORCH_HOME"] = str(cache_root / "torch")
    return environment


if win32serviceutil is not None:

    class CellVisionProductionService(win32serviceutil.ServiceFramework):
        _svc_name_ = SERVICE_NAME
        _svc_display_name_ = SERVICE_DISPLAY_NAME
        _svc_description_ = (
            "Runs the shared Cell Vision project queue and inference engine "
            "independently of interactive domain-user sessions."
        )

        def __init__(self, args: list[str]):
            super().__init__(args)
            self.stop_event = win32event.CreateEvent(None, 0, 0, None)
            self.child: subprocess.Popen[Any] | None = None

        def SvcStop(self) -> None:  # noqa: N802
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            win32event.SetEvent(self.stop_event)
            self._stop_child()

        def _stop_child(self) -> None:
            child = self.child
            if child is None or child.poll() is not None:
                return
            subprocess.run(
                ["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)),
            )

        def SvcDoRun(self) -> None:  # noqa: N802
            root = installation_root()
            values = load_production_environment(root)
            environment = service_process_environment(root, values)
            log_root = Path(values.get("CELLVISION_LOG_ROOT", str(root / "logs")))
            log_root.mkdir(parents=True, exist_ok=True)
            command = production_command(root, values)
            with (log_root / "cellvision-service.stdout.log").open("a", encoding="utf-8") as stdout, (
                log_root / "cellvision-service.stderr.log"
            ).open("a", encoding="utf-8") as stderr:
                self.child = subprocess.Popen(
                    command,
                    cwd=str(root),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)),
                )
                servicemanager.LogInfoMsg(
                    f"{SERVICE_DISPLAY_NAME} started child PID {self.child.pid}"
                )
                while True:
                    if win32event.WaitForSingleObject(self.stop_event, 1000) == win32event.WAIT_OBJECT_0:
                        self._stop_child()
                        return
                    exit_code = self.child.poll()
                    if exit_code is not None:
                        message = f"Cell Vision production child exited unexpectedly: {exit_code}"
                        servicemanager.LogErrorMsg(message)
                        raise RuntimeError(message)

else:

    class CellVisionProductionService:  # pragma: no cover
        pass


def wait_for_service_stop(seconds: float = 1.0) -> None:
    time.sleep(max(0.0, seconds))
