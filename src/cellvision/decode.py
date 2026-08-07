from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image


def quick_hash(path: str | Path, block_size: int = 65536) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    size = source.stat().st_size
    with source.open("rb") as handle:
        digest.update(handle.read(block_size))
        if size > block_size:
            handle.seek(max(0, size - block_size))
            digest.update(handle.read(block_size))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def inspect_tiff(path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "width_px": "",
        "height_px": "",
        "bit_depth": "",
        "channels": "",
        "pyramid_levels": "",
        "decode_status": "error",
        "decode_error": "",
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            mode_to_bits = {"1": 1, "L": 8, "I;16": 16, "I": 32, "F": 32}
            result.update(
                width_px=image.width,
                height_px=image.height,
                bit_depth=mode_to_bits.get(image.mode, 8),
                channels=len(image.getbands()),
                pyramid_levels=getattr(image, "n_frames", 1),
                decode_status="ok",
            )
    except Exception as exc:  # data audit must record and continue
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_grayscale(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").copy()


def read_mask(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("1").copy()

