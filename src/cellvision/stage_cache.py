from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stage_fingerprint(
    name: str,
    paths: Iterable[str | Path],
    settings: Any,
    *,
    version: str,
) -> str:
    files = []
    for raw_path in sorted({str(Path(path).resolve()) for path in paths}):
        path = Path(raw_path)
        if path.exists():
            stat = path.stat()
            files.append(
                {
                    "path": raw_path,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        else:
            files.append({"path": raw_path, "missing": True})
    payload = {
        "name": name,
        "version": version,
        "files": files,
        "settings": settings,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
