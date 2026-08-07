from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from .config import artifact_path
from .manifest import build_manifest


def _nvidia_info() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "error": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        name, total, free, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {
            "available": True,
            "name": name,
            "memory_total_mb": int(total),
            "memory_free_mb": int(free),
            "driver_version": driver,
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    images, sequences = build_manifest(config)
    data_root = Path(config["paths"]["data_root"])
    disk = shutil.disk_usage(data_root)
    unreadable_rows = images[
        (images["decode_status"] != "ok") | (images["cf_decode_status"] != "ok")
    ]
    unreadable = unreadable_rows[
        [
            "well",
            "timepoint",
            "raw_image_path",
            "cf_image_path",
            "decode_error",
            "cf_decode_error",
        ]
    ].to_dict(orient="records")
    unreadable_file_count = int(
        (unreadable_rows["decode_status"] != "ok").sum()
        + (unreadable_rows["cf_decode_status"] != "ok").sum()
    )
    audit = {
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
            "data_disk_free_gb": round(disk.free / 1024**3, 2),
            "nvidia": _nvidia_info(),
        },
        "data": {
            "data_root": str(data_root),
            "data_root_online": data_root.exists(),
            "plate_count": int(images["plate_id"].nunique()),
            "well_count": int(sequences.shape[0]),
            "image_rows": int(images.shape[0]),
            "complete_t0_t2_sequences": int((sequences["sequence_status"] == "complete_t0_t2").sum()),
            "unreadable_or_missing_sequence_rows": len(unreadable),
            "unreadable_or_missing_file_count": unreadable_file_count,
            "unreadable_or_missing": unreadable,
            "raw_dimensions": images[images["decode_status"] == "ok"]
            .groupby(["width_px", "height_px", "bit_depth"])
            .size()
            .reset_index(name="count")
            .to_dict(orient="records"),
            "resolution_um_per_pixel": config["calibration"]["resolution_um_per_pixel"],
            "calibration_source": "session.dat ResolutionMMPerPixel=0.00208, verified manually",
        },
        "training_readiness": {
            "human_object_annotations": 1,
            "supervised_classification_ready": False,
            "weak_cf_segmentation_ready": True,
            "reason": "Only one human Single cell annotation exists; CF masks are usable only as weak labels.",
        },
    }
    path = artifact_path(config, "system_audit.json")
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit
