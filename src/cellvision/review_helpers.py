from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, signal
from skimage import measure

from .decode import inspect_tiff


def _review_images_manifest(
    config: dict[str, Any], base: pd.DataFrame
) -> pd.DataFrame:
    """Append Day7/Day14 images for review without expanding inference scope."""

    directories = config.get("review", {}).get("late_timepoint_directories", {})
    if not directories:
        return base
    exp = config["experiment"]
    extra: list[dict[str, Any]] = []
    for timepoint, directory in directories.items():
        if str(timepoint).upper() not in {"T3", "T4"}:
            continue
        folder = Path(str(directory))
        for row_name in "ABCDEFGH":
            for column in range(1, 13):
                well = f"{row_name}{column}"
                raw = folder / f"{well}.tif"
                cf = folder / f"{well}-cf.tif"
                metadata = inspect_tiff(raw) if raw.exists() else {
                    "width_px": 0,
                    "height_px": 0,
                    "bit_depth": 0,
                    "channels": 0,
                    "pyramid_levels": 0,
                    "decode_status": "missing",
                    "decode_error": "raw image missing",
                }
                extra.append({
                    "experiment_id": exp["experiment_id"],
                    "plate_id": exp["plate_id"],
                    "well": well,
                    "timepoint": str(timepoint).upper(),
                    "raw_image_path": str(raw.resolve()) if raw.exists() else "",
                    "cf_image_path": str(cf.resolve()) if cf.exists() else "",
                    "metrics_csv_path": str((folder / "metricsummary.csv").resolve()),
                    "cf_decode_status": "ok" if cf.exists() else "missing",
                    **metadata,
                })
    if not extra:
        return base
    combined = pd.concat([base, pd.DataFrame(extra)], ignore_index=True, sort=False)
    combined["well"] = combined["well"].astype(str).str.upper()
    combined["timepoint"] = combined["timepoint"].astype(str).str.upper()
    return combined.drop_duplicates(["well", "timepoint"], keep="last")


def _growth_region_contours(
    mask_path: Path,
    metrics_path: Path,
    well: str,
    settings: dict[str, Any],
    *,
    raw_image_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract thick Day14 sheet contours while rejecting thin wall bands."""

    if not mask_path.exists():
        return []
    downsample = max(1, int(settings.get("downsample", 4)))
    minimum_coverage = float(settings.get("minimum_component_coverage_pct", 1.0))
    minimum_radius = float(settings.get("minimum_radius_px", 40.0))
    minimum_mean_distance = float(settings.get("minimum_mean_distance_px", 10.0))
    with Image.open(mask_path) as opened:
        width = max(1, opened.width // downsample)
        height = max(1, opened.height // downsample)
        mask = np.asarray(
            opened.resize((width, height), Image.Resampling.NEAREST).convert("L")
        ) > 0
    # The CF image often contains the complete circular well rim.  Its very
    # long connected arc can merge with a real colony and make the displayed
    # growth region wrap around the wall.  Estimate the innermost strong
    # circular edge from the raw overview and clip the CF mask to the well
    # interior before connected-component analysis.  This affects only the
    # explanatory overlay, not the trained cell/debris detector.
    if raw_image_path is not None and raw_image_path.exists() and mask.any():
        try:
            with Image.open(raw_image_path) as opened:
                raw = np.asarray(
                    opened.resize((width, height), Image.Resampling.BILINEAR).convert("L"),
                    dtype=np.float32,
                )
            gradient = np.hypot(ndimage.sobel(raw, axis=0), ndimage.sobel(raw, axis=1))
            yy, xx = np.indices(mask.shape)
            center_y = (height - 1) / 2.0
            center_x = (width - 1) / 2.0
            radius_map = np.hypot(xx - center_x, yy - center_y)
            radial_index = np.floor(radius_map).astype(np.int32)
            radial_sum = np.bincount(radial_index.ravel(), weights=gradient.ravel())
            radial_count = np.bincount(radial_index.ravel())
            radial_mean = radial_sum / np.maximum(radial_count, 1)
            radial_mean = ndimage.gaussian_filter1d(radial_mean, sigma=2.0)
            minimum_radius = int(min(width, height) * 0.30)
            maximum_radius = int(min(width, height) * 0.49)
            search = radial_mean[minimum_radius : maximum_radius + 1]
            if search.size:
                peak_floor = float(np.percentile(search, 88))
                peaks, properties = signal.find_peaks(
                    search,
                    height=peak_floor,
                    distance=max(2, int(min(width, height) * 0.008)),
                )
                if len(peaks):
                    # Prefer the inner edge among similarly strong rim peaks.
                    peak_heights = properties["peak_heights"]
                    strong = peaks[peak_heights >= 0.72 * float(peak_heights.max())]
                    rim_radius = float(minimum_radius + int(strong.min()))
                    inner_margin = max(2.0, min(width, height) * 0.006)
                    mask &= radius_map <= rim_radius - inner_margin
        except (OSError, ValueError):
            pass
    foreground = int(mask.sum())
    if foreground == 0:
        return []
    confluence = 0.0
    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            selected = metrics[metrics["Well"].astype(str).str.upper() == well.upper()]
            if not selected.empty:
                confluence = float(selected.iloc[0]["Cell Confluence"])
        except (KeyError, OSError, ValueError):
            confluence = 0.0
    effective_area = (
        foreground / max(confluence / 100.0, 1e-9)
        if confluence > 0
        else np.pi * (min(width, height) * 0.39) ** 2
    )
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    areas = np.bincount(labels.ravel())[1:]
    objects = ndimage.find_objects(labels)
    contours: list[dict[str, Any]] = []
    for index in np.argsort(areas)[-min(len(areas), 48):][::-1]:
        coverage = float(areas[index] / effective_area * 100.0)
        if coverage < minimum_coverage:
            continue
        slices = objects[index]
        if slices is None:
            continue
        component = labels[slices] == index + 1
        padded_component = np.pad(component, 1, mode="constant")
        distance = ndimage.distance_transform_edt(padded_component)[1:-1, 1:-1]
        radius = float(distance.max() * downsample)
        mean_distance = float(distance[component].mean() * downsample)
        if radius < minimum_radius or mean_distance < minimum_mean_distance:
            continue
        padded = padded_component.astype(np.uint8)
        candidates = measure.find_contours(padded, 0.5)
        if not candidates:
            continue
        contour = max(candidates, key=len)
        stride = max(1, int(np.ceil(len(contour) / 260)))
        sampled = contour[::stride]
        points = [
            [
                float((point[1] - 1 + slices[1].start) * downsample),
                float((point[0] - 1 + slices[0].start) * downsample),
            ]
            for point in sampled
        ]
        if len(points) >= 3:
            contours.append({
                "points": points,
                "coverage_pct": coverage,
                "maximum_radius_px": radius,
            })
    return contours


def _boolean_series(values: pd.Series) -> pd.Series:
    """Return a real boolean mask even after heterogeneous row concatenation.

    Appending manual-review rows changes pandas boolean columns to ``object``.
    Applying ``~`` to Python bools stored in an object series produces -1/-2,
    which are both truthy and therefore accidentally exposes suppressed V2
    proposals.  Normalize explicitly before any visibility filtering.
    """

    def convert(value: Any) -> bool:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    return values.map(convert).astype(bool)


FINAL_REASON_TEXTS = {
    "stable_wall_site_structure": (
        "三帧可靠匹配；目标持续位于孔壁，形态稳定且无分裂或生物变化，"
        "判定为孔壁结构伪目标。"
    ),
    "division_or_growth_vetoes_static_debris_and_dead_cell": (
        "检测到分裂或增长，分裂证据优先，保留为细胞并撤销静态杂质/死细胞推断。"
    ),
    "cross_track_division_rescue_vetoes_static_debris_and_dead_cell": (
        "相邻轨迹形成可信分裂关系，分裂证据优先，保留为细胞。"
    ),
    "strong_t0_cell_monotonic_decline_with_morphology_degradation": (
        "T0 为强细胞证据，随后细胞置信与形态连续退化，整条轨迹统一判定为死细胞。"
    ),
    "three_frame_stable_noncell_without_persistent_cell_evidence": (
        "三帧可靠匹配且形态稳定，没有持续细胞或分裂证据，判定为杂质。"
    ),
    "three_frame_morphology_stable_noncell_without_persistent_cell_evidence": (
        "三帧形态稳定（允许明暗变化），没有持续细胞或分裂证据，判定为杂质。"
    ),
    "cell_probability_decline_without_morphology_degradation": (
        "细胞置信下降，但没有同步形态退化，证据不足以判定为死细胞。"
    ),
    "stable_track_has_strong_cell_evidence_no_debris_override": (
        "三帧轨迹稳定，同时存在持续强细胞证据，时序规则不改判为杂质。"
    ),
    "strong_multiframe_cell_evidence_overrides_wall_structure": (
        "至少两帧具有高置信细胞形态且无效概率低，强细胞证据否决孔壁结构误检。"
    ),
    "wall_object_requires_biological_branch": (
        "候选位于孔壁，但具有重复紧致形态或生物变化，按独立目标分类。"
    ),
    "wall_site_insufficient_structure_evidence": (
        "目标位于孔壁，但孔壁结构证据或跨帧证据不足，需要人工复核。"
    ),
    "no_decisive_temporal_behavior": (
        "时序证据没有达到改判阈值，最终分类沿用识别模型结论。"
    ),
}


def _decision_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


CELL_SUBTYPE_LABELS = {"single", "touching_doublet", "cluster_3plus"}


def _cell_subtype_for_track_review(
    row: pd.Series,
    *,
    prefer_frame_review: bool,
    fallback: str = "single",
) -> str:
    """Resolve a cell-family track review without flattening multiplicity.

    A track-level review answers whether every member is a biological cell.
    Single/doublet/3+ remains a frame-level observation and may legitimately
    change after division.  Legacy track reviews stored an exact cell subtype;
    for those rows prefer the model's per-frame subtype so an old unified
    ``single`` review no longer forces every frame back to one cell.
    """

    reviewed = _decision_text(row.get("reviewed_label"))
    current = _decision_text(row.get("current_label"))
    proposed = _decision_text(row.get("v3_proposed_label"))
    adjusted = _decision_text(row.get("v2_temporal_adjusted_label"))
    integrated = _decision_text(row.get("integrated_label"))
    pre_temporal = _decision_text(row.get("v2_pre_temporal_integrated_label"))
    candidates = (
        [reviewed, current, proposed, adjusted, integrated, pre_temporal]
        if prefer_frame_review
        else [proposed, adjusted, integrated, pre_temporal, reviewed, current]
    )
    for label in candidates:
        if label in CELL_SUBTYPE_LABELS:
            return label

    probabilities = {
        "single": _decision_number(row.get("single_probability"), 0.0),
        "touching_doublet": _decision_number(
            row.get("touching_doublet_probability"), 0.0
        ),
        "cluster_3plus": _decision_number(
            row.get("cluster_3plus_probability"), 0.0
        ),
    }
    if any(value > 0.0 for value in probabilities.values()):
        return max(probabilities, key=probabilities.get)
    return fallback if fallback in CELL_SUBTYPE_LABELS else "single"


def _final_review_label(value: Any) -> str:
    """Map the authoritative conclusion to a label supported by frame review."""

    label = _decision_text(value) or "uncertain"
    return "debris" if label == "dead_cell" else label


def _decision_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(number) else number


def _decision_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if np.isfinite(value) else False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _with_final_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose one authoritative classification for each review object.

    Raw frame probabilities, legacy appearance matching and V2/V3 evidence
    remain available for auditing, but they must not compete with the result
    shown to the reviewer.  This contract applies a single priority order:
    human track review, human frame review, V3 unified/temporal decision, V2
    adjustment, then the integrated frame model.
    """

    if frame.empty:
        return frame.copy()
    result = frame.copy()
    final_labels: list[str] = []
    final_sources: list[str] = []
    final_reason_codes: list[str] = []
    final_reason_texts: list[str] = []
    final_confidences: list[float] = []
    final_statuses: list[str] = []

    for _, row in result.iterrows():
        current_label = (
            _decision_text(row.get("current_label"))
            or _decision_text(row.get("integrated_label"))
            or "uncertain"
        )
        reviewed_label = _decision_text(row.get("reviewed_label"))
        track_review = _decision_text(row.get("v3_reviewed_label"))
        track_behavior = _decision_text(row.get("v3_track_behavior"))
        track_conclusion = (
            _decision_text(row.get("v3_track_conclusion"))
            or _decision_text(row.get("v3_unified_label"))
        )
        proposed_label = _decision_text(row.get("v3_proposed_label"))
        v3_reason = _decision_text(row.get("v3_reason"))
        integrated_confidence = _decision_number(
            row.get("integrated_confidence"), 0.0
        )
        behavior_confidence = _decision_number(
            row.get("v3_behavior_score"), integrated_confidence
        )

        if track_review == "cell" or track_review in CELL_SUBTYPE_LABELS:
            final_label = _cell_subtype_for_track_review(
                row,
                prefer_frame_review=track_review == "cell",
                fallback=track_review,
            )
            source = "human_track_review"
            reason_code = "human_track_cell_review"
            reason_text = (
                "人工已确认整条时序轨迹均属于细胞；"
                "单细胞、黏连2细胞和3+细胞团保留逐帧结论。"
            )
            confidence = 1.0
        elif track_review and track_review != "unmarked":
            final_label = track_review
            source = "human_track_review"
            reason_code = "human_track_review"
            reason_text = "人工已统一审核整条时序轨迹，覆盖模型结论。"
            confidence = 1.0
        elif reviewed_label:
            final_label = reviewed_label
            source = "human_frame_review"
            reason_code = "human_frame_review"
            reason_text = "人工已审核当前目标，覆盖模型结论。"
            confidence = 1.0
        elif track_conclusion and _decision_text(row.get("v3_label_mode")) == "unified_track":
            final_label = track_conclusion
            source = "v3_unified_track"
            reason_code = v3_reason or track_behavior
            reason_text = FINAL_REASON_TEXTS.get(
                reason_code, "V3 已根据整条轨迹给出统一分类。"
            )
            confidence = behavior_confidence
        elif track_behavior in {
            "no_decisive_temporal_evidence",
            "wall_uncertain",
            "decline_without_morphology_evidence",
        }:
            final_label = current_label
            source = "integrated_model"
            reason_code = v3_reason or track_behavior
            reason_text = FINAL_REASON_TEXTS.get(
                reason_code,
                "时序证据未达到改判阈值，最终分类沿用识别模型结论。",
            )
            confidence = integrated_confidence
        elif track_behavior and track_behavior != "disabled" and proposed_label:
            final_label = proposed_label
            source = "v3_temporal"
            reason_code = v3_reason or track_behavior
            reason_text = FINAL_REASON_TEXTS.get(
                reason_code, "V3 已综合多帧身份、形态与事件证据给出分类。"
            )
            confidence = behavior_confidence
        elif (
            _decision_bool(row.get("v2_temporal_adjustment_applied"))
            and _decision_text(row.get("v2_temporal_adjusted_label"))
        ):
            final_label = _decision_text(row.get("v2_temporal_adjusted_label"))
            source = "v2_temporal"
            reason_code = _decision_text(row.get("v2_temporal_reason"))
            reason_text = "V2 时序证据达到改判阈值，已覆盖单帧分类。"
            confidence = integrated_confidence
        else:
            final_label = current_label
            source = "integrated_model"
            reason_code = "integrated_model_result"
            reason_text = "时序证据未触发改判，最终分类沿用识别模型结论。"
            confidence = integrated_confidence

        confidence = float(np.clip(confidence, 0.0, 1.0))
        needs_review = bool(
            final_label == "uncertain"
            or track_behavior == "wall_uncertain"
        )
        final_labels.append(final_label)
        final_sources.append(source)
        final_reason_codes.append(reason_code)
        final_reason_texts.append(reason_text)
        final_confidences.append(confidence)
        final_statuses.append("needs_review" if needs_review else "determined")

    result["final_label"] = final_labels
    result["final_review_label"] = [
        _final_review_label(label) for label in final_labels
    ]
    result["final_source"] = final_sources
    result["final_reason_code"] = final_reason_codes
    result["final_reason_text"] = final_reason_texts
    result["final_confidence"] = final_confidences
    result["final_status"] = final_statuses
    return result


def _visible_v2_review_instances(reviewable: pd.DataFrame) -> pd.DataFrame:
    """Keep one authoritative review row for each V2 instance."""

    manual = _boolean_series(
        reviewable.get("is_manual_missed", pd.Series(False, index=reviewable.index))
    )
    mask_valid = _boolean_series(reviewable["v2_mask_valid"])
    wall_rejected = _boolean_series(reviewable["v2_wall_rejected"])
    suppressed = _boolean_series(reviewable["v2_is_suppressed"])
    visible = reviewable[manual | (mask_valid & ~wall_rejected & ~suppressed)].copy()

    # Suppression should already guarantee uniqueness.  This final guard keeps
    # the API invariant stable if an older result file contains two owner rows
    # with the same instance id.
    instance_id = visible["v2_instance_id"].fillna("").astype(str)
    has_instance = ~instance_id.isin(["", "nan", "None"])
    with_instance = visible[has_instance].copy()
    without_instance = visible[~has_instance].copy()
    if not with_instance.empty:
        with_instance["_reviewed_priority"] = with_instance.get(
            "reviewed_label", pd.Series(None, index=with_instance.index)
        ).notna().astype(int)
        with_instance["_instance_confidence_priority"] = pd.to_numeric(
            with_instance.get(
                "v2_instance_confidence", pd.Series(0.0, index=with_instance.index)
            ),
            errors="coerce",
        ).fillna(0.0)
        with_instance = (
            with_instance.sort_values(
                ["_reviewed_priority", "_instance_confidence_priority"],
                ascending=False,
                kind="stable",
            )
            .drop_duplicates(["well", "timepoint", "v2_instance_id"], keep="first")
            .drop(columns=["_reviewed_priority", "_instance_confidence_priority"])
        )
    return pd.concat([with_instance, without_instance], axis=0).sort_index()


