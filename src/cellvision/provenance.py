"""Run provenance helpers for reproducible artifact directories."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _path_fingerprint(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for value in paths:
        path = Path(value).expanduser()
        try:
            stat = path.stat()
            rows.append({
                "path": str(path.resolve()),
                "exists": True,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        except OSError:
            rows.append({"path": str(path.resolve()), "exists": False})
    return rows


def create_run_metadata(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    input_paths: Iterable[str | Path] = (),
    model_paths: Iterable[str | Path] = (),
    status: str = "running",
    stages: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write/update a small metadata record beside a run's results."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "run_metadata.json"
    existing: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    metadata = {
        **existing,
        "run_id": existing.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "config_digest": _json_digest(config),
        "input_fingerprints": _path_fingerprint(input_paths) if input_paths else existing.get("input_fingerprints", []),
        "model_fingerprints": _path_fingerprint(model_paths) if model_paths else existing.get("model_fingerprints", []),
        "code_revision": _git_revision(Path(__file__).resolve().parents[2]),
        "stages": stages or existing.get("stages", []),
    }
    if extra:
        metadata.update(extra)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata_path
