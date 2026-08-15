from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from .v2_instance_dataset import _crop
from .v2_instance_inference import decode_rle


TIMEPOINT_ORDER = {"T0": 0, "T1": 1, "T2": 2}
TRACK_BEHAVIORS = (
    "cell_to_debris",
    "division_or_growth",
    "stable_debris",
    "wall_structure_invalid",
    "wall_independent_object",
    "wall_uncertain",
    "stable_cell_or_conflict",
    "decline_without_morphology_evidence",
    "no_decisive_temporal_evidence",
)
REVIEW_DECISIONS = {"accept_v3", "keep_legacy", "manual_labels", "needs_more", "skip"}
FRAME_LABELS = {"single", "touching_doublet", "cluster_3plus", "debris", "invalid", "uncertain", "unmarked"}
FRAME_LABELS.add("dead_cell")
TRACK_LABELS = FRAME_LABELS.copy()
UNIFIED_CELL_TO_DEBRIS_LABELS = {"dead_cell", "uncertain", "debris", "invalid", "unmarked"}

# These are review thresholds, not new inference thresholds.  They surface a
# three-frame track that was accepted into the object graph but does not clear
# the morphology-stability gates comfortably enough for blind acceptance.
LOW_MATCH_REVIEW_THRESHOLDS = {
    "identity": 0.70,
    "static": 0.68,
    "shape": 0.86,
    "foreground_quality": 0.50,
}


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def _text(row: pd.Series, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return str(value)


def _number(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _boolean(row: pd.Series, key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _conditional_cell(row: pd.Series) -> float:
    cell = max(_number(row, "cell_probability"), 0.0)
    debris = max(_number(row, "debris_probability"), 0.0)
    invalid = max(_number(row, "invalid_probability"), 0.0)
    return float(cell / max(cell + debris + invalid, 1e-8))


def _safe_run_id(root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid shadow run id")
    path = (root / run_id).resolve()
    if not path.is_relative_to(root.resolve()):
        raise HTTPException(status_code=400, detail="Shadow run is outside the configured root")
    if not (path / "report.json").exists():
        raise HTTPException(status_code=404, detail=f"Shadow run not found: {run_id}")
    return path


def _read_report(run_path: Path) -> dict[str, Any]:
    try:
        return json.loads((run_path / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read shadow report: {exc}") from exc


def _plate_output_path(run_path: Path, plate: dict[str, Any]) -> Path:
    raw = Path(str(plate.get("output_predictions", ""))).expanduser().resolve()
    if not raw.is_relative_to(run_path.resolve()):
        raise HTTPException(status_code=400, detail="Shadow prediction path is outside the run directory")
    if not raw.exists():
        raise HTTPException(status_code=404, detail=f"Prediction file not found: {raw}")
    return raw


@lru_cache(maxsize=8)
def _load_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _review_path(run_path: Path) -> Path:
    return run_path / "temporal_shadow_reviews.json"


def _review_key(plate_id: str, track_id: str) -> str:
    return f"{plate_id}::{track_id}"


def _read_reviews(run_path: Path) -> dict[str, dict[str, Any]]:
    path = _review_path(run_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read review file: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _write_reviews(run_path: Path, reviews: dict[str, dict[str, Any]]) -> None:
    path = _review_path(run_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _frame_record(row: pd.Series) -> dict[str, Any]:
    conditional = _conditional_cell(row)
    return {
        "candidate_id": _text(row, "candidate_id"),
        "timepoint": _text(row, "timepoint"),
        "well": _text(row, "well"),
        "legacy_label": _text(row, "v2_pre_temporal_integrated_label", _text(row, "integrated_label")),
        "legacy_final_label": _text(row, "integrated_label"),
        "v2_final_label": _text(row, "integrated_label"),
        "v2_temporal_label": _text(row, "v2_temporal_adjusted_label"),
        "v3_label": _text(row, "v3_proposed_label", _text(row, "integrated_label")),
        "v3_track_conclusion": _text(row, "v3_track_conclusion"),
        "v3_unified_label": _text(row, "v3_unified_label"),
        "v3_label_mode": _text(row, "v3_label_mode", "per_frame_evidence"),
        "v3_frame_state": _text(row, "v3_frame_state", "preserved"),
        "cell_probability": _number(row, "cell_probability"),
        "debris_probability": _number(row, "debris_probability"),
        "invalid_probability": _number(row, "invalid_probability"),
        # Kept for API compatibility; this is now invalid-aware full-class
        # cell evidence rather than cell-vs-debris-only probability.
        "conditional_cell_probability": conditional,
        "cell_evidence_probability": conditional,
        "v3_would_change": _boolean(row, "v3_would_change"),
        "semantic_degradation": _boolean(row, "v3_semantic_degradation"),
        "morphology_change_score": _number(row, "v3_morphology_change_score"),
        "x_px": _number(row, "x_px"),
        "y_px": _number(row, "y_px"),
        "raw_image_path": _text(row, "raw_image_path"),
        "candidate_source": _text(row, "candidate_source"),
        "radial_fraction": _number(row, "radial_fraction"),
        "wall_overlap": _number(row, "v2_wall_overlap"),
        "wall_neighbor_count": _number(row, "wall_neighbor_count"),
        "review_anisotropy": _number(row, "review_anisotropy"),
        "background_anisotropy": _number(row, "background_anisotropy"),
        "wall_rescue_blobness": _number(row, "wall_rescue_blobness"),
        "instance_confidence": _number(row, "v2_instance_confidence"),
        "objectness": _number(row, "v2_objectness"),
        "v2_temporal_same_object_score": _number(row, "v2_temporal_same_object_score"),
        "v2_temporal_static_similarity": _number(row, "v2_temporal_static_similarity_score"),
        "v2_temporal_shape_similarity": _number(row, "v2_temporal_shape_similarity"),
        "v2_temporal_foreground_quality": _number(row, "v2_temporal_foreground_quality"),
        "v2_temporal_candidate_count": int(_number(row, "v2_temporal_candidate_count")),
        "v2_temporal_pair_count": int(_number(row, "v2_temporal_pair_count")),
    }


def _track_record(
    plate_id: str,
    group: pd.DataFrame,
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    ordered = group.assign(
        _time_order=group["timepoint"].astype(str).map(TIMEPOINT_ORDER).fillna(9)
    ).sort_values(["_time_order", "candidate_id"])
    first = ordered.iloc[0]
    frames = [_frame_record(row) for _, row in ordered.iterrows()]
    timepoints = {
        timepoint: [frame for frame in frames if frame["timepoint"] == timepoint]
        for timepoint in ("T0", "T1", "T2")
    }
    changed = bool(any(frame["v3_would_change"] for frame in frames))
    behavior = _text(first, "v3_track_behavior", "")
    track_conclusion = _text(first, "v3_track_conclusion")
    # Shadow runs generated before the unified-conclusion column existed are
    # still safe to review: infer the new track-level proposal from the
    # already established behavior without rewriting the CSV.
    if behavior == "cell_to_debris" and not track_conclusion:
        track_conclusion = "dead_cell"
    v2_v3_mismatches = [
        frame
        for frame in frames
        if frame["v2_final_label"] != frame["v3_label"]
    ]
    unified_conclusion_difference = bool(
        behavior == "cell_to_debris" and track_conclusion
    )
    v2_v3_mismatch_reasons = []
    if v2_v3_mismatches:
        v2_v3_mismatch_reasons.append("frame_label_difference")
    if unified_conclusion_difference:
        v2_v3_mismatch_reasons.append("v3_unified_track_conclusion")
    v2_v3_mismatch = bool(v2_v3_mismatch_reasons)
    matched_three_frames = all(timepoints[timepoint] for timepoint in ("T0", "T1", "T2"))
    identity_score = _number(first, "v3_identity_score")
    static_similarity = _number(first, "v3_static_similarity")
    shape_similarity = _number(first, "v3_shape_similarity")
    foreground_quality = _number(first, "v3_foreground_quality")
    low_match_reasons: list[str] = []
    if matched_three_frames and len(frames) >= 3:
        if identity_score < LOW_MATCH_REVIEW_THRESHOLDS["identity"]:
            low_match_reasons.append("identity")
        if static_similarity < LOW_MATCH_REVIEW_THRESHOLDS["static"]:
            low_match_reasons.append("static_similarity")
        if shape_similarity < LOW_MATCH_REVIEW_THRESHOLDS["shape"]:
            low_match_reasons.append("shape_similarity")
        if foreground_quality < LOW_MATCH_REVIEW_THRESHOLDS["foreground_quality"]:
            low_match_reasons.append("foreground_quality")
    low_match_confidence = bool(low_match_reasons)
    legacy_review_needs_unification = bool(
        behavior == "cell_to_debris"
        and review is not None
        and not str(review.get("track_label", "")).strip()
    )
    priority = {
        "cell_to_debris": 0,
        "division_or_growth": 1,
        "wall_structure_invalid": 2,
        "stable_debris": 3,
        "wall_independent_object": 4,
        "wall_uncertain": 5,
        "stable_cell_or_conflict": 6,
        "decline_without_morphology_evidence": 7,
        "no_decisive_temporal_evidence": 8,
    }.get(behavior, 9)
    return {
        "plate_id": plate_id,
        "track_id": _text(first, "v3_track_id"),
        "well": _text(first, "well"),
        "behavior": behavior,
        "track_conclusion": track_conclusion,
        "requires_unified_label": behavior == "cell_to_debris",
        "legacy_review_needs_unification": legacy_review_needs_unification,
        "wall_origin": _text(first, "v3_wall_origin", "none"),
        "reason": _text(first, "v3_reason"),
        "behavior_score": _number(first, "v3_behavior_score"),
        "identity_score": identity_score,
        "static_similarity": static_similarity,
        "shape_similarity": shape_similarity,
        "morphology_change_score": _number(first, "v3_morphology_change_score"),
        "degradation_evidence_score": _number(first, "v3_degradation_evidence_score"),
        "semantic_degradation": _boolean(first, "v3_semantic_degradation"),
        "division_veto": _boolean(first, "v3_division_veto"),
        "division_interval": _text(first, "v3_division_interval"),
        "frame_count": len(frames),
        "changed": changed,
        "v2_v3_mismatch": v2_v3_mismatch,
        "v2_v3_mismatch_count": len(v2_v3_mismatches),
        "v2_v3_mismatch_reasons": v2_v3_mismatch_reasons,
        "matched_three_frames": matched_three_frames,
        "low_match_confidence": low_match_confidence,
        "low_match_reasons": low_match_reasons,
        "foreground_quality": foreground_quality,
        "priority": priority,
        "reviewed": review is not None,
        "review": review,
        "frames": frames,
        "timepoints": timepoints,
    }


def _all_plate_ids(report: dict[str, Any]) -> list[str]:
    return [str(plate.get("plate_id", "")) for plate in report.get("plates", [])]


def _find_plate(report: dict[str, Any], plate_id: str) -> dict[str, Any]:
    for plate in report.get("plates", []):
        if str(plate.get("plate_id", "")) == plate_id:
            return plate
    raise HTTPException(status_code=404, detail=f"Plate not found: {plate_id}")


@lru_cache(maxsize=16)
def _base_plate_tracks(run_path_string: str, plate_id: str) -> list[dict[str, Any]]:
    run_path = Path(run_path_string)
    report = _read_report(run_path)
    plate = _find_plate(report, plate_id)
    predictions = _load_predictions(str(_plate_output_path(run_path, plate)))
    if "v3_track_behavior" not in predictions:
        return []
    visible = predictions[predictions["v3_track_behavior"].astype(str) != "disabled"].copy()
    tracks: list[dict[str, Any]] = []
    for track_id, group in visible.groupby("v3_track_id", sort=False):
        track_id = str(track_id)
        tracks.append(_track_record(plate_id, group, None))
    tracks.sort(key=lambda item: (item["priority"], not item["changed"], item["well"], item["track_id"]))
    return tracks


def _plate_tracks(
    run_path: Path,
    report: dict[str, Any],
    plate_id: str,
    reviews: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    base_tracks = _base_plate_tracks(str(run_path), plate_id)
    result = []
    for base in base_tracks:
        review = reviews.get(_review_key(plate_id, base["track_id"]))
        item = base.copy()
        item["reviewed"] = review is not None
        item["review"] = review
        item["legacy_review_needs_unification"] = bool(
            item["requires_unified_label"]
            and review is not None
            and not str(review.get("track_label", "")).strip()
        )
        result.append(item)
    return result


def _focus_matches(track: dict[str, Any], focus: str) -> bool:
    if focus == "all":
        return True
    if focus == "v2_v3_diff":
        return bool(track["v2_v3_mismatch"])
    if focus == "low_match_confidence":
        return bool(track["low_match_confidence"])
    if focus == "focus":
        return bool(track["v2_v3_mismatch"] or track["low_match_confidence"])
    raise HTTPException(status_code=422, detail=f"Invalid focus queue: {focus}")


def _well_summaries(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for track in tracks:
        key = (str(track["plate_id"]), str(track["well"]))
        grouped.setdefault(key, []).append(track)
    result: list[dict[str, Any]] = []
    for (plate_id, well), well_tracks in sorted(grouped.items()):
        result.append(
            {
                "plate_id": plate_id,
                "well": well,
                "track_count": len(well_tracks),
                "v2_v3_mismatch_track_count": sum(
                    bool(track["v2_v3_mismatch"]) for track in well_tracks
                ),
                "low_match_track_count": sum(
                    bool(track["low_match_confidence"]) for track in well_tracks
                ),
                "legacy_changed_track_count": sum(
                    bool(track["changed"]) for track in well_tracks
                ),
                "reviewed_track_count": sum(
                    bool(track["reviewed"]) for track in well_tracks
                ),
                "behaviors": sorted({str(track["behavior"]) for track in well_tracks}),
                "v2_labels": sorted(
                    {
                        frame["v2_final_label"]
                        for track in well_tracks
                        for frame in track["frames"]
                    }
                ),
                "v3_labels": sorted(
                    {
                        frame["v3_label"]
                        for track in well_tracks
                        for frame in track["frames"]
                    }
                ),
                "v2_v3_mismatch_reasons": sorted(
                    {
                        reason
                        for track in well_tracks
                        for reason in track["v2_v3_mismatch_reasons"]
                    }
                ),
                "low_match_reasons": sorted(
                    {
                        reason
                        for track in well_tracks
                        for reason in track["low_match_reasons"]
                    }
                ),
            }
        )
    return result


class TrackReviewPayload(BaseModel):
    run_id: str
    plate_id: str
    track_id: str
    decision: str = Field(default="needs_more")
    reviewer: str = Field(default="local_user", max_length=120)
    notes: str = Field(default="", max_length=4000)
    track_label: str = Field(default="", max_length=40)
    frame_labels: dict[str, str] = Field(default_factory=dict)


def create_shadow_review_app(root: str | Path) -> FastAPI:
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    ui_root = Path(__file__).resolve().parents[2] / "review-ui"
    review_file_lock = threading.Lock()
    app = FastAPI(title="CellVision Temporal Shadow Review")
    app.mount("/assets", StaticFiles(directory=ui_root), name="shadow-review-assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (ui_root / "temporal-shadow-review.html").read_text(encoding="utf-8")

    @app.get("/api/shadow/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "root": str(root_path), "runs": len(list(root_path.glob("*/report.json")))}

    @app.get("/api/shadow/runs")
    def runs() -> list[dict[str, Any]]:
        result = []
        for report_path in sorted(root_path.glob("*/report.json"), reverse=True):
            run_path = report_path.parent
            report = _read_report(run_path)
            result.append(
                {
                    "run_id": run_path.name,
                    "mode": report.get("mode", "shadow"),
                    "checkpoint": report.get("checkpoint", ""),
                    "plates": _all_plate_ids(report),
                    "modified_at": datetime.fromtimestamp(
                        report_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        # Prefer the active run covering the most plates.  The directory also
        # contains one-plate diagnostic runs which should not silently replace
        # the 20-plate review queue.
        result.sort(
            key=lambda item: (
                item["mode"] == "v3_active",
                len(item["plates"]),
                item["modified_at"],
            ),
            reverse=True,
        )
        return result

    @app.get("/api/shadow/overview")
    def overview(run_id: str) -> dict[str, Any]:
        run_path = _safe_run_id(root_path, run_id)
        report = _read_report(run_path)
        reviews = _read_reviews(run_path)
        plate_summaries = []
        behavior_counts: dict[str, int] = {}
        total_tracks = 0
        reviewed_tracks = 0
        changed_tracks = 0
        all_tracks: list[dict[str, Any]] = []
        for plate_id in _all_plate_ids(report):
            tracks = _plate_tracks(run_path, report, plate_id, reviews)
            all_tracks.extend(tracks)
            total_tracks += len(tracks)
            reviewed_tracks += sum(track["reviewed"] for track in tracks)
            changed_tracks += sum(track["changed"] for track in tracks)
            counts = pd.Series([track["behavior"] for track in tracks]).value_counts().to_dict()
            for behavior, count in counts.items():
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + int(count)
            plate_summaries.append(
                {
                    "plate_id": plate_id,
                    "track_count": len(tracks),
                    "reviewed_count": sum(track["reviewed"] for track in tracks),
                    "changed_count": sum(track["changed"] for track in tracks),
                    "v2_v3_mismatch_count": sum(
                        track["v2_v3_mismatch"] for track in tracks
                    ),
                    "low_match_count": sum(
                        track["low_match_confidence"] for track in tracks
                    ),
                    "focus_well_count": len(
                        _well_summaries(
                            [
                                track
                                for track in tracks
                                if _focus_matches(track, "focus")
                            ]
                        )
                    ),
                    "behavior_counts": counts,
                }
            )
        mismatch_wells = _well_summaries(
            [track for track in all_tracks if _focus_matches(track, "v2_v3_diff")]
        )
        low_match_wells = _well_summaries(
            [track for track in all_tracks if _focus_matches(track, "low_match_confidence")]
        )
        focus_wells = _well_summaries(
            [track for track in all_tracks if _focus_matches(track, "focus")]
        )
        return {
            "run_id": run_id,
            "mode": report.get("mode", "shadow"),
            "total_tracks": total_tracks,
            "reviewed_tracks": reviewed_tracks,
            "changed_tracks": changed_tracks,
            "v2_v3_mismatch_tracks": sum(track["v2_v3_mismatch"] for track in all_tracks),
            "low_match_tracks": sum(track["low_match_confidence"] for track in all_tracks),
            "focus_tracks": len(
                [track for track in all_tracks if _focus_matches(track, "focus")]
            ),
            "v2_v3_mismatch_wells": len(mismatch_wells),
            "low_match_wells": len(low_match_wells),
            "focus_wells": len(focus_wells),
            "focus_thresholds": LOW_MATCH_REVIEW_THRESHOLDS,
            "behavior_counts": behavior_counts,
            "plates": plate_summaries,
        }

    @app.get("/api/shadow/wells")
    def wells(
        run_id: str,
        plate_id: str = "",
        focus: Literal["all", "v2_v3_diff", "low_match_confidence", "focus"] = "focus",
    ) -> dict[str, Any]:
        run_path = _safe_run_id(root_path, run_id)
        report = _read_report(run_path)
        reviews = _read_reviews(run_path)
        selected_plates = [plate_id] if plate_id else _all_plate_ids(report)
        selected: list[dict[str, Any]] = []
        for selected_plate in selected_plates:
            for track in _plate_tracks(run_path, report, selected_plate, reviews):
                if _focus_matches(track, focus):
                    selected.append(track)
        return {
            "run_id": run_id,
            "focus": focus,
            "thresholds": LOW_MATCH_REVIEW_THRESHOLDS,
            "wells": _well_summaries(selected),
        }

    @app.get("/api/shadow/tracks")
    def tracks(
        run_id: str,
        plate_id: str = "",
        behavior: str = "all",
        focus: Literal["all", "v2_v3_diff", "low_match_confidence", "focus"] = "focus",
        changed_only: bool = False,
        review_status: Literal["all", "pending", "reviewed"] = "all",
        search: str = "",
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=60, ge=1, le=200),
    ) -> dict[str, Any]:
        run_path = _safe_run_id(root_path, run_id)
        report = _read_report(run_path)
        reviews = _read_reviews(run_path)
        selected_plates = [plate_id] if plate_id else _all_plate_ids(report)
        selected: list[dict[str, Any]] = []
        needle = search.strip().lower()
        for selected_plate in selected_plates:
            for track in _plate_tracks(run_path, report, selected_plate, reviews):
                if behavior != "all" and track["behavior"] != behavior:
                    continue
                if not _focus_matches(track, focus):
                    continue
                if changed_only and not track["changed"]:
                    continue
                if review_status == "pending" and track["reviewed"]:
                    continue
                if review_status == "reviewed" and not track["reviewed"]:
                    continue
                if needle and needle not in f"{track['well']} {track['track_id']} {track['reason']}".lower():
                    continue
                selected.append(track)
        total = len(selected)
        start = (page - 1) * page_size
        return {
            "run_id": run_id,
            "total": total,
            "page": page,
            "page_size": page_size,
            "tracks": selected[start : start + page_size],
        }

    @app.get("/api/shadow/track")
    def track(run_id: str, plate_id: str, track_id: str) -> dict[str, Any]:
        run_path = _safe_run_id(root_path, run_id)
        report = _read_report(run_path)
        reviews = _read_reviews(run_path)
        for item in _plate_tracks(run_path, report, plate_id, reviews):
            if item["track_id"] == track_id:
                return item
        raise HTTPException(status_code=404, detail="Track not found")

    @app.get("/api/shadow/image")
    def image(
        run_id: str,
        plate_id: str,
        candidate_id: str,
        size: int = Query(default=240, ge=96, le=512),
        mask: bool = True,
    ) -> Response:
        run_path = _safe_run_id(root_path, run_id)
        report = _read_report(run_path)
        plate = _find_plate(report, plate_id)
        predictions = _load_predictions(str(_plate_output_path(run_path, plate)))
        matches = predictions[predictions["candidate_id"].astype(str) == candidate_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail="Candidate not found")
        row = matches.iloc[0]
        raw_path = Path(_text(row, "raw_image_path"))
        if not raw_path.exists():
            raise HTTPException(status_code=404, detail=f"Raw image not found: {raw_path}")
        with Image.open(raw_path) as opened:
            raw = np.asarray(opened.convert("L"), dtype=np.uint8)
        crop = _crop(raw, _number(row, "x_px"), _number(row, "y_px"), size, int(np.median(raw)))
        values = crop.astype(np.float32)
        low, high = np.percentile(values, [2, 98])
        values = np.clip((values - low) / max(float(high - low), 1.0), 0.0, 1.0)
        rendered = Image.fromarray(np.asarray(values * 255.0, dtype=np.uint8), mode="L").convert("RGB")
        draw = ImageDraw.Draw(rendered)
        if mask:
            try:
                decoded = decode_rle(_text(row, "v2_mask_rle", "[]"), 96)
                mask_image = Image.fromarray((decoded.astype(np.uint8) * 150), mode="L")
                offset = (size - 96) // 2
                overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                overlay_color = Image.new("RGBA", (96, 96), (41, 211, 145, 0))
                overlay_color.putalpha(mask_image)
                overlay.alpha_composite(overlay_color, (offset, offset))
                rendered = Image.alpha_composite(rendered.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(rendered)
            except (ValueError, TypeError):
                pass
        draw.rectangle((0, 0, size - 1, size - 1), outline=(255, 255, 255), width=2)
        buffer = BytesIO()
        rendered.save(buffer, format="JPEG", quality=90)
        return Response(content=buffer.getvalue(), media_type="image/jpeg")

    @app.get("/api/shadow/reviews")
    def reviews(run_id: str) -> dict[str, Any]:
        run_path = _safe_run_id(root_path, run_id)
        return {"run_id": run_id, "reviews": _read_reviews(run_path)}

    @app.post("/api/shadow/review")
    def save_review(payload: TrackReviewPayload) -> dict[str, Any]:
        run_path = _safe_run_id(root_path, payload.run_id)
        if payload.decision not in REVIEW_DECISIONS:
            raise HTTPException(status_code=422, detail=f"Invalid review decision: {payload.decision}")
        report = _read_report(run_path)
        reviews = _read_reviews(run_path)
        track_data = None
        for item in _plate_tracks(run_path, report, payload.plate_id, reviews):
            if item["track_id"] == payload.track_id:
                track_data = item
                break
        if track_data is None:
            raise HTTPException(status_code=404, detail="Track not found")
        candidate_ids = {frame["candidate_id"] for frame in track_data["frames"]}
        invalid_ids = set(payload.frame_labels) - candidate_ids
        if invalid_ids:
            raise HTTPException(status_code=422, detail=f"Frame labels contain unknown candidates: {sorted(invalid_ids)}")
        invalid_labels = set(payload.frame_labels.values()) - FRAME_LABELS
        if invalid_labels:
            raise HTTPException(status_code=422, detail=f"Invalid frame labels: {sorted(invalid_labels)}")
        track_label = payload.track_label.strip()
        if track_label and track_label not in TRACK_LABELS:
            raise HTTPException(status_code=422, detail=f"Invalid track label: {track_label}")
        saved_frame_labels = dict(payload.frame_labels)
        if track_data["requires_unified_label"]:
            if not track_label:
                track_label = track_data["track_conclusion"] or "dead_cell"
            if track_label not in UNIFIED_CELL_TO_DEBRIS_LABELS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "cell_to_debris tracks require one unified track label: "
                        f"{sorted(UNIFIED_CELL_TO_DEBRIS_LABELS)}"
                    ),
                )
            # The biological event is the annotation unit.  Keep the frame
            # IDs for traceability, but never persist a mixed T0/T1/T2 label
            # sequence for this branch.
            saved_frame_labels = {
                frame["candidate_id"]: track_label
                for frame in track_data["frames"]
            }
        record = {
            "run_id": payload.run_id,
            "plate_id": payload.plate_id,
            "track_id": payload.track_id,
            "decision": payload.decision,
            "reviewer": payload.reviewer.strip() or "local_user",
            "notes": payload.notes,
            "track_label": track_label,
            "frame_labels": saved_frame_labels,
            "label_mode": "unified_track" if track_data["requires_unified_label"] else "per_frame",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with review_file_lock:
            reviews = _read_reviews(run_path)
            reviews[_review_key(payload.plate_id, payload.track_id)] = record
            _write_reviews(run_path, reviews)
        return {"ok": True, "review": record}

    return app
