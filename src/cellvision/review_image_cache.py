from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageEnhance

from .config import artifact_path


def review_image_cache_path(config: dict[str, Any], raw_path: str | Path, max_size: int) -> tuple[Path, str]:
    source = Path(raw_path)
    identity = f"{source}|{source.stat().st_mtime_ns}|{max_size}|contrast=1.7"
    etag = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return artifact_path(config, "cache", "review_images", f"{etag}.jpg"), etag


def render_review_image(config: dict[str, Any], raw_path: str | Path, max_size: int) -> Path:
    source = Path(raw_path)
    output, _ = review_image_cache_path(config, source, max_size)
    if output.exists():
        return output
    with Image.open(source) as image:
        gray = ImageEnhance.Contrast(image.convert("L")).enhance(1.7)
        gray.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        rendered = gray.copy()
    temporary = output.with_suffix(".tmp")
    rendered.save(temporary, format="JPEG", quality=94 if max_size > 1800 else 90, subsampling=0)
    temporary.replace(output)
    return output


def precache_review_images(config: dict[str, Any], sizes: tuple[int, ...] = (1400,)) -> dict[str, int]:
    manifest = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    manifest = manifest[manifest["decode_status"].eq("ok") & manifest["timepoint"].isin(["T0", "T1", "T2"])]
    tasks = [(raw_path, size) for raw_path in manifest["raw_image_path"].astype(str).unique() for size in sizes]
    existing = sum(review_image_cache_path(config, path, size)[0].exists() for path, size in tasks)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda task: render_review_image(config, task[0], task[1]), tasks))
    return {"requested": len(tasks), "already_cached": int(existing), "generated": int(len(tasks) - existing)}
