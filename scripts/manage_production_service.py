"""Install or control the Cell Vision pywin32 service host."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import win32serviceutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pywin32 is not installed in the machine service runtime") from exc

from cellvision.windows_service import CellVisionProductionService


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(CellVisionProductionService)
