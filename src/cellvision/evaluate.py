from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run_metrics(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))

