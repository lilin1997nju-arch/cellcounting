from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


ROWS = "ABCDEFGH"
COLUMNS = range(1, 13)
PLATE_WELLS = tuple(f"{row}{column}" for row in ROWS for column in COLUMNS)

CATEGORY_LABELS = {
    "positive_control": "阳性对照",
    "no_obvious_growth": "无明显生长",
    "single_cell_origin": "单细胞来源",
    "multi_cell_origin": "多细胞来源",
    "undetermined": "待确定",
}

REASON_LABELS = {
    "": "",
    "day14_negative": "Day14无明显成片生长",
    "day14_missing_or_unreadable": "Day14缺失或图像不可用",
    "early_results_not_computed": "尚未完成T0～T2深度计算",
    "early_images_incomplete": "T0～T2图像不完整",
    "image_quality_failure": "图像质量不足",
    "t0_classification_uncertain": "T0细胞/杂质归类不确定",
    "t0_missing_later_detected": "T0缺失但后期出现细胞",
    "day14_growth_but_no_early_cell": "Day14有生长但T0～T2均未检出细胞",
    "t0_t2_no_division": "T0-T2未分裂",
    "temporal_link_conflict": "早期时序关联冲突",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "retain", "positive"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def classify_gated_well(
    *,
    day14_available: bool,
    day14_obvious_growth: bool,
    is_positive_control: bool = False,
    early_results_available: bool = True,
    early_timepoints_complete: bool = True,
    image_quality_failure: bool = False,
    t0_cell_units: int = 0,
    t1_cell_units: int = 0,
    t2_cell_units: int = 0,
    t0_cell_instances: int = 0,
    t0_has_uncertain: bool = False,
    temporal_link_conflict: bool = False,
    early_division_evidence: bool | None = None,
) -> dict[str, Any]:
    """Make one mutually exclusive well conclusion.

    Day14 is a compute gate, not an origin classifier.  A negative Day14 well
    never enters the expensive early-timepoint path.  A positive well is
    classified solely from T0-T2 evidence; Day7 is intentionally absent from
    this function because it only supplies a representative visual region.
    """

    if is_positive_control:
        return _decision("positive_control", "", skip=True)
    if not day14_available:
        return _decision("undetermined", "day14_missing_or_unreadable", skip=True)
    if not day14_obvious_growth:
        return _decision("no_obvious_growth", "day14_negative", skip=True)
    if not early_results_available:
        return _decision("undetermined", "early_results_not_computed", skip=False)
    if not early_timepoints_complete:
        return _decision("undetermined", "early_images_incomplete", skip=False)
    if image_quality_failure:
        return _decision("undetermined", "image_quality_failure", skip=False)
    if t0_has_uncertain:
        return _decision("undetermined", "t0_classification_uncertain", skip=False)

    t0_cell_units = max(0, int(t0_cell_units))
    t1_cell_units = max(0, int(t1_cell_units))
    t2_cell_units = max(0, int(t2_cell_units))
    t0_cell_instances = max(0, int(t0_cell_instances))

    if t0_cell_units == 0:
        reason = (
            "t0_missing_later_detected"
            if max(t1_cell_units, t2_cell_units) > 0
            else "day14_growth_but_no_early_cell"
        )
        return _decision("undetermined", reason, skip=False)

    # A touching doublet contributes two units and a 3+ cluster contributes at
    # least three. Multiple independent T0 instances are also multi-origin.
    if t0_cell_units > 1 or t0_cell_instances > 1:
        return _decision("multi_cell_origin", "", skip=False)

    if temporal_link_conflict:
        return _decision("undetermined", "temporal_link_conflict", skip=False)

    # The unit counts are the authoritative multiplicity signal.  Do not let
    # a stale/low-confidence boolean hide a T1/T2 touching doublet or cluster
    # after a single T0 origin.
    early_division_evidence = bool(early_division_evidence) or max(
        t1_cell_units, t2_cell_units
    ) >= 2
    if bool(early_division_evidence):
        return _decision("single_cell_origin", "", skip=False)
    return _decision("undetermined", "t0_t2_no_division", skip=False)


def _decision(category: str, reason: str, *, skip: bool) -> dict[str, Any]:
    return {
        "final_category": category,
        "final_category_label": CATEGORY_LABELS[category],
        "undetermined_reason": reason,
        "undetermined_reason_label": REASON_LABELS[reason],
        "skip_early_computation": bool(skip),
        "requires_manual_review": category == "undetermined",
    }


def locate_day7_dense_regions(
    cf_mask_path: str | Path,
    *,
    roi_size_px: int = 1000,
    max_regions: int = 3,
    max_dimension: int = 1024,
) -> dict[str, Any]:
    """Locate representative Day7 cell-dense windows without cell inference.

    The high-contrast CF image is reduced to at most 1024 px, then scored with
    an integral-equivalent box filter.  The score combines total foreground
    and a thickness-aware foreground map, so isolated texture contributes
    little while a dense sheet is retained.  No object label or temporal
    correction is produced.
    """

    path = Path(cf_mask_path)
    if not path.exists():
        return {"available": False, "method": "missing_cf", "regions": []}
    with Image.open(path) as image:
        mask_full = np.asarray(image.convert("L")) > 0
    height, width = mask_full.shape
    scale = min(1.0, float(max_dimension) / float(max(height, width)))
    small_width = max(1, int(round(width * scale)))
    small_height = max(1, int(round(height * scale)))
    if scale < 1.0:
        mask = np.asarray(
            Image.fromarray(mask_full.astype(np.uint8) * 255).resize(
                (small_width, small_height), Image.Resampling.NEAREST
            )
        ) > 0
    else:
        mask = mask_full

    # Remove single-pixel acquisition noise but preserve small real objects.
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    window = max(8, int(round(roi_size_px * scale)))
    window = min(window, small_height, small_width)
    density = ndimage.uniform_filter(mask.astype(np.float32), size=window, mode="constant")
    distance = ndimage.distance_transform_edt(mask)
    thick = distance >= max(1.5, 3.0 * scale)
    thick_density = ndimage.uniform_filter(thick.astype(np.float32), size=window, mode="constant")
    score = 0.65 * density + 0.35 * thick_density

    half = window // 2
    valid = np.zeros_like(mask, dtype=bool)
    valid[half : max(half + 1, small_height - half), half : max(half + 1, small_width - half)] = True
    if not valid.any():
        valid[:] = True
    work = np.where(valid, score, -np.inf)
    regions: list[dict[str, Any]] = []
    suppression_radius = max(1, int(round(window * 0.65)))
    for _ in range(max(1, int(max_regions))):
        flat = int(np.argmax(work))
        best = float(work.flat[flat])
        if not np.isfinite(best) or best <= 0:
            break
        y_small, x_small = np.unravel_index(flat, work.shape)
        x = float(x_small / scale)
        y = float(y_small / scale)
        x0 = max(0.0, min(float(width - roi_size_px), x - roi_size_px / 2.0))
        y0 = max(0.0, min(float(height - roi_size_px), y - roi_size_px / 2.0))
        x1 = min(float(width), x0 + roi_size_px)
        y1 = min(float(height), y0 + roi_size_px)
        regions.append({
            "rank": len(regions) + 1,
            "center_x": float((x0 + x1) / 2.0),
            "center_y": float((y0 + y1) / 2.0),
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "foreground_fraction": float(density[y_small, x_small]),
            "thick_foreground_fraction": float(thick_density[y_small, x_small]),
            "score": best,
        })
        yy0 = max(0, y_small - suppression_radius)
        yy1 = min(small_height, y_small + suppression_radius + 1)
        xx0 = max(0, x_small - suppression_radius)
        xx1 = min(small_width, x_small + suppression_radius + 1)
        work[yy0:yy1, xx0:xx1] = -np.inf

    return {
        "available": True,
        "method": "downsampled_cf_density_no_temporal",
        "image_width": int(width),
        "image_height": int(height),
        "scale": float(scale),
        "roi_size_px": int(roi_size_px),
        "regions": regions,
    }


def build_gated_plate_report(
    day14_csv: str | Path,
    group_id: str,
    output_dir: str | Path,
    *,
    early_screening_csv: str | Path | None = None,
    sessions_csv: str | Path | None = None,
    locate_day7: bool = False,
    day14_growth_overrides: dict[str, str] | None = None,
    endpoint_day_label: str = "Day14",
) -> dict[str, Any]:
    """Build an endpoint-gated queue and the final 96-well overview data.

    ``day14_csv`` is retained as a compatibility name for existing QL2603
    artifacts.  The contents can now describe any selected late culture day;
    ``endpoint_day_label`` records which actual day was used for the gate.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # The former Day7 density locator could not reliably identify where later
    # growth originated.  It is intentionally disabled even when an older
    # caller passes locate_day7=True; stale regions are not carried forward.
    # The previous-timepoint image remains available as a full-well view.
    _ = (locate_day7, sessions_csv)
    endpoint = pd.read_csv(day14_csv, low_memory=False)
    endpoint = endpoint[endpoint["group_id"].astype(str) == str(group_id)].copy()
    if endpoint.empty:
        raise ValueError(f"{endpoint_day_label} screening has no rows for group: {group_id}")
    endpoint["well"] = endpoint["well"].astype(str).str.upper()
    endpoint_lookup = endpoint.drop_duplicates("well", keep="last").set_index("well")
    normalized_day14_overrides = {
        str(well).upper(): str(decision)
        for well, decision in (day14_growth_overrides or {}).items()
    }

    early_lookup: dict[str, dict[str, Any]] = {}
    if early_screening_csv:
        early_path = Path(early_screening_csv)
        if early_path.exists():
            early = pd.read_csv(early_path, low_memory=False)
            if "group_id" in early.columns:
                early = early[early["group_id"].astype(str) == str(group_id)]
            early["well"] = early["well"].astype(str).str.upper()
            early_lookup = early.drop_duplicates("well", keep="last").set_index("well").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    for well in PLATE_WELLS:
        endpoint_row = endpoint_lookup.loc[well].to_dict() if well in endpoint_lookup.index else {}
        # Accept both the new endpoint column names and the original Day14
        # screening names, allowing old trained artifacts to be reused.
        day14_row = endpoint_row
        early_row = early_lookup.get(well, {})
        is_control = _as_bool(day14_row.get("is_positive_control"), well == "A1")
        endpoint_available = bool(endpoint_row) and bool(endpoint_row.get("raw_image_path"))
        endpoint_positive = _as_bool(
            endpoint_row.get("endpoint_obvious_sheet_growth", endpoint_row.get("day14_obvious_sheet_growth")),
            str(endpoint_row.get("screening_decision", "")).lower() == "retain",
        )
        day14_override = normalized_day14_overrides.get(well, "")
        if day14_override == "obvious_growth":
            endpoint_positive = True
        elif day14_override == "no_growth":
            endpoint_positive = False
        day14_available = endpoint_available
        day14_positive = endpoint_positive
        early_available = bool(early_row)
        t0 = _as_int(early_row.get("t0_cell_units"))
        t1 = _as_int(early_row.get("t1_cell_units"))
        t2 = _as_int(early_row.get("t2_cell_units"))
        t0_instances = _as_int(
            early_row.get("t0_cell_instances"),
            0 if t0 == 0 else 1 if t0 == 1 else t0,
        )
        division = early_row.get("early_division_evidence")
        if division is None or (not isinstance(division, (dict, list)) and pd.isna(division)):
            division_value: bool | None = None
        else:
            division_value = _as_bool(division)
        decision = classify_gated_well(
            day14_available=day14_available,
            day14_obvious_growth=day14_positive,
            is_positive_control=is_control,
            early_results_available=early_available,
            early_timepoints_complete=_as_bool(early_row.get("early_timepoints_complete"), True),
            image_quality_failure=_as_bool(early_row.get("image_quality_failure"), False),
            t0_cell_units=t0,
            t1_cell_units=t1,
            t2_cell_units=t2,
            t0_cell_instances=t0_instances,
            t0_has_uncertain=_as_bool(early_row.get("t0_has_uncertain"), False),
            temporal_link_conflict=_as_bool(early_row.get("temporal_link_conflict"), False),
            early_division_evidence=division_value,
        )
        if str(endpoint_day_label) != "Day14":
            decision["undetermined_reason_label"] = str(
                decision.get("undetermined_reason_label", "")
            ).replace("Day14", str(endpoint_day_label))

        day7_result: dict[str, Any] = {
            "available": False,
            "method": "disabled_full_well_only",
            "regions": [],
        }

        evidence_notes: list[str] = []
        if _as_bool(early_row.get("has_debris"), False):
            evidence_notes.append("存在杂质")

        rows.append({
            "group_id": group_id,
            "well": well,
            "is_positive_control": is_control,
            "day14_available": day14_available,
            "day14_obvious_growth": day14_positive,
            "endpoint_day_label": str(endpoint_day_label),
            "endpoint_available": endpoint_available,
            "endpoint_obvious_growth": endpoint_positive,
            "day14_screening_decision": (
                f"human_{day14_override}"
                if day14_override in {"obvious_growth", "no_growth"}
                else day14_row.get("screening_decision", "missing")
            ),
            "day14_instrument_confluence_pct": day14_row.get("instrument_confluence_pct", np.nan),
            "day14_sheet_coverage_pct": day14_row.get("sheet_coverage_pct", np.nan),
            "day14_maximum_sheet_radius_px": day14_row.get("maximum_sheet_radius_px", np.nan),
            "day14_raw_image_path": day14_row.get("raw_image_path", ""),
            "day14_cf_mask_path": day14_row.get("cf_mask_path", ""),
            "early_results_available": early_available,
            "t0_cell_units": t0,
            "t1_cell_units": t1,
            "t2_cell_units": t2,
            "t0_cell_instances": t0_instances,
            "early_division_evidence": division_value if division_value is not None else max(t1, t2) >= 2,
            "has_debris": _as_bool(early_row.get("has_debris"), False),
            # The V2 dead-cell signal is retired.  Retain the field as a
            # backwards-compatible false value for existing CSV consumers.
            "suspected_dead_cell": False,
            **decision,
            "day7_available": bool(day7_result.get("available")),
            "day7_method": day7_result.get("method", ""),
            "day7_regions_json": json.dumps(day7_result.get("regions", []), ensure_ascii=False),
            "evidence_notes": "；".join(evidence_notes),
        })

    result = pd.DataFrame(rows)
    report_csv = output / "plate_overview.csv"
    report_json = output / "plate_overview.json"
    queue_csv = output / "day14_positive_early_compute_queue.csv"
    result.to_csv(report_csv, index=False, encoding="utf-8")
    queue = result[
        result["day14_obvious_growth"]
        & ~result["is_positive_control"]
    ].copy()
    queue.to_csv(queue_csv, index=False, encoding="utf-8")
    category_counts = result["final_category"].value_counts().to_dict()
    reason_counts = (
        result.loc[result["final_category"] == "undetermined", "undetermined_reason"]
        .value_counts()
        .to_dict()
    )
    payload = {
        "group_id": group_id,
        "well_count": int(len(result)),
        "sample_well_count": int((~result["is_positive_control"]).sum()),
        "day14_positive_sample_wells": int(len(queue)),
        "day14_skipped_sample_wells": int(
            ((result["final_category"] == "no_obvious_growth") & ~result["is_positive_control"]).sum()
        ),
        "category_counts": {str(key): int(value) for key, value in category_counts.items()},
        "undetermined_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "report_csv": str(report_csv.resolve()),
        "report_json": str(report_json.resolve()),
        "early_compute_queue_csv": str(queue_csv.resolve()),
        "endpoint_day_label": str(endpoint_day_label),
        "wells": result.to_dict(orient="records"),
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
