from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageDraw, ImageEnhance

from .config import artifact_path, load_config
from .review_image_cache import (
    patch_cache_path,
    render_patch,
    render_review_image,
    report_image_cache_path,
    review_image_cache_path,
)
from .well_screening import build_well_screening


def register_image_routes(
    app: FastAPI,
    config: dict[str, Any],
    images_manifest: pd.DataFrame,
    database: str | Path,
) -> None:
    """Register the review image-serving routes (patch / well / report)."""

    @app.get("/api/patch")
    def patch(
        request: Request,
        well: str,
        timepoint: str,
        x: float,
        y: float,
        size: int = 256,
        source_config: str | None = None,
    ) -> Response:
        image_manifest = images_manifest
        if source_config:
            try:
                comparison_config = load_config(source_config)
                comparison_manifest = (
                    Path(comparison_config["paths"]["artifact_root"])
                    / "manifests"
                    / "images.csv"
                )
                if comparison_manifest.exists():
                    image_manifest = pd.read_csv(comparison_manifest)
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, pd.errors.ParserError):
                image_manifest = None
        if image_manifest is None:
            raise HTTPException(status_code=503, detail="image manifest unavailable")
        selected = image_manifest[
            (image_manifest["well"] == well.upper())
            & (image_manifest["timepoint"] == timepoint.upper())
            & (image_manifest["decode_status"] == "ok")
        ]
        if selected.empty:
            raise HTTPException(status_code=404, detail="image unavailable")
        size = max(64, min(int(size), 2048))
        raw_path = Path(str(selected.iloc[0]["raw_image_path"]))
        cache_path, etag = patch_cache_path(config, raw_path, x, y, size)
        headers = {"ETag": f'"{etag}"', "Cache-Control": "public, max-age=31536000, immutable"}
        if request.headers.get("if-none-match") == headers["ETag"]:
            return Response(status_code=304, headers=headers)
        if not cache_path.exists():
            render_patch(config, raw_path, x, y, size)
        return Response(content=cache_path.read_bytes(), media_type="image/jpeg", headers=headers)

    @app.get("/api/well-image")
    def well_image(
        request: Request, well: str, timepoint: str, max_size: int = 1200
    ) -> Response:
        selected = images_manifest[
            (images_manifest["well"] == well.upper())
            & (images_manifest["timepoint"] == timepoint.upper())
            & (images_manifest["decode_status"] == "ok")
        ]
        if selected.empty:
            raise HTTPException(status_code=404, detail="image unavailable")
        max_size = max(400, min(int(max_size), 4096))
        raw_path = Path(str(selected.iloc[0]["raw_image_path"]))
        cache_path, etag = review_image_cache_path(config, raw_path, max_size)
        headers = {"ETag": f'"{etag}"', "Cache-Control": "public, max-age=31536000, immutable"}
        if request.headers.get("if-none-match") == headers["ETag"]:
            return Response(status_code=304, headers=headers)
        if not cache_path.exists():
            render_review_image(config, raw_path, max_size)
        return Response(content=cache_path.read_bytes(), media_type="image/jpeg", headers=headers)

    @app.get("/api/report-image")
    def report_image(
        request: Request,
        well: str,
        timepoint: str,
        view: str = "whole",
        max_size: int = 1200,
    ) -> Response:
        if view not in {"whole", "local"}:
            raise HTTPException(status_code=422, detail="invalid view")
        normalized_well = well.upper()
        normalized_timepoint = timepoint.upper()
        selected = images_manifest[
            (images_manifest["well"] == normalized_well)
            & (images_manifest["timepoint"] == normalized_timepoint)
            & (images_manifest["decode_status"] == "ok")
        ]
        if selected.empty:
            raise HTTPException(status_code=404, detail="image unavailable")
        screening_path = artifact_path(
            config, "predictions", "latest_well_screening.csv"
        )
        if not screening_path.exists():
            build_well_screening(config, database)
        screening = pd.read_csv(screening_path)
        row = screening[screening["well"].astype(str).str.upper() == normalized_well]
        if row.empty:
            raise HTTPException(status_code=404, detail="screening ROI unavailable")
        roi_map = json.loads(str(row.iloc[0]["roi_json"]))
        roi = roi_map.get(normalized_timepoint)
        raw_path = Path(str(selected.iloc[0]["raw_image_path"]))
        max_size = max(360, min(int(max_size), 2200))
        cache_path: Path | None = None
        headers: dict[str, str] = {}
        if roi is not None:
            cache_path, etag = report_image_cache_path(
                config,
                raw_path,
                float(roi["x"]),
                float(roi["y"]),
                float(roi["size"]),
                view,
                max_size,
            )
            headers = {"ETag": f'"{etag}"', "Cache-Control": "public, max-age=31536000, immutable"}
            if request.headers.get("if-none-match") == headers["ETag"]:
                return Response(status_code=304, headers=headers)
            if cache_path.exists():
                return Response(content=cache_path.read_bytes(), media_type="image/jpeg", headers=headers)
        with Image.open(raw_path) as opened:
            rendered = ImageEnhance.Contrast(opened.convert("L")).enhance(1.7)
            width, height = rendered.size
            if roi is None:
                roi = {"x": width / 2, "y": height / 2, "size": min(width, height) * 0.45}
            size = max(240, min(int(float(roi["size"])), min(width, height)))
            half = size // 2
            x = float(roi["x"])
            y = float(roi["y"])
            left = max(0, min(width - size, int(round(x - half))))
            top = max(0, min(height - size, int(round(y - half))))
            right, bottom = left + size, top + size
            if view == "local":
                # The report crop is intentionally clean: no object circles.
                rendered = rendered.crop((left, top, right, bottom))
            else:
                draw = ImageDraw.Draw(rendered)
                line_width = max(8, min(width, height) // 350)
                draw.rectangle(
                    (left, top, right, bottom),
                    outline=255,
                    width=line_width,
                )
            rendered.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            rendered.save(temporary, format="JPEG", quality=94, subsampling=0)
            temporary.replace(cache_path)
            return Response(content=cache_path.read_bytes(), media_type="image/jpeg", headers=headers)
        buffer = io.BytesIO()
        rendered.save(buffer, format="JPEG", quality=94, subsampling=0)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

