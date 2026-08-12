from __future__ import annotations

import json
import hashlib
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter, gaussian_filter1d, label as ndi_label, zoom
from skimage.morphology import closing, dilation, disk, erosion, remove_small_holes, remove_small_objects
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
from skimage.measure import find_contours, regionprops

from .config import artifact_path
from .models.v2_instance_segmenter import SeededInstanceUNet
from .v2_instance_dataset import _crop, _seed_heatmap, _wall_prior


LOW_CELL_NONCELL_RESOLUTION_THRESHOLD = 0.30


def _normalize(raw: np.ndarray) -> np.ndarray:
    value = raw.astype(np.float32)
    lo, hi = np.percentile(value, [2, 98])
    return np.clip((value - lo) / max(float(hi - lo), 1.0), 0, 1)


def _selected_component(
    binary: np.ndarray,
    seed_weights: np.ndarray | None = None,
    *,
    retain_nearby: bool = False,
    nearby_distance: int = 3,
) -> np.ndarray:
    """Select the component supported by the seed, optionally retaining group lobes.

    Instance crops are seed-conditioned, so component ownership should follow
    the seed heatmap rather than only the component centroid.  Reviewed
    doublets and clusters may contain two thresholded lobes separated by a
    narrow one- or two-pixel gap; those nearby lobes are reconnected before
    contour extraction instead of being silently discarded.
    """

    binary = np.asarray(binary, dtype=bool)
    labelled, _ = ndi_label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    regions = regionprops(labelled)
    if not regions:
        return np.zeros_like(binary, dtype=bool)
    center = np.asarray([(binary.shape[0] - 1) / 2, (binary.shape[1] - 1) / 2])
    eligible = [region for region in regions if region.area >= 3]
    if not eligible:
        return np.zeros_like(binary, dtype=bool)
    if seed_weights is not None:
        weights = np.asarray(seed_weights, dtype=np.float32)
        if weights.shape != binary.shape:
            raise ValueError("seed_weights must have the same shape as binary")
        chosen = max(
            eligible,
            key=lambda region: (
                float(weights[labelled == region.label].sum()),
                -float(np.linalg.norm(np.asarray(region.centroid) - center)),
                float(region.area),
            ),
        )
    else:
        chosen = min(
            eligible,
            key=lambda region: np.linalg.norm(np.asarray(region.centroid) - center),
        )
    selected = labelled == chosen.label
    if not retain_nearby:
        return selected

    remaining = [region for region in eligible if region.label != chosen.label]
    # Grow iteratively because a three-cell cluster may form a short chain.
    while remaining:
        neighborhood = dilation(selected, disk(max(int(nearby_distance), 1)))
        attached = [region for region in remaining if np.any(neighborhood & (labelled == region.label))]
        if not attached:
            break
        for region in attached:
            selected |= labelled == region.label
            remaining.remove(region)
    if selected.any():
        selected = closing(selected, disk(min(max(int(nearby_distance), 1), 2)))
    return selected


def _rle(mask: np.ndarray) -> str:
    values = mask.astype(np.uint8).ravel()
    padded = np.pad(values, (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    runs = changes.reshape(-1, 2)
    return json.dumps([[int(start), int(end - start)] for start, end in runs], separators=(",", ":"))


def decode_rle(value: str, size: int) -> np.ndarray:
    mask = np.zeros(size * size, dtype=bool)
    for start, length in json.loads(value or "[]"):
        mask[int(start) : int(start) + int(length)] = True
    return mask.reshape(size, size)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / max(int(union), 1))


def refine_instance_mask(
    mask: np.ndarray,
    raw: np.ndarray,
    wall_probability: np.ndarray,
    seed_weights: np.ndarray | None = None,
    *,
    retain_nearby: bool = False,
    nearby_distance: int = 3,
    minimum_area_ratio: float = 0.80,
    minimum_iou: float = 0.70,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Snap a coarse mask to nearby image edges without allowing large shape drift."""
    original = remove_small_objects(mask.astype(bool), max_size=2)
    original = closing(original, disk(1))
    # Cell interiors often contain a dark nucleus or phase halo.  A closed
    # cavity is therefore foreground for whole-cell feature extraction, not a
    # second boundary to be preserved.
    original = binary_fill_holes(original)
    if not original.any():
        if diagnostics is not None:
            diagnostics.update(status="empty", area_ratio=0.0, iou=0.0)
        return original
    edge_map = inverse_gaussian_gradient(raw.astype(np.float32), alpha=80.0, sigma=1.0)
    evolved = morphological_geodesic_active_contour(
        edge_map, 7, init_level_set=original, smoothing=1, threshold="auto", balloon=0
    ).astype(bool)
    allowed = dilation(original, disk(3))
    protected_core = erosion(original, disk(2))
    evolved = (evolved & allowed) | protected_core
    # A real cell may overlap the wall, but new contour growth is not allowed
    # to run longitudinally into very high-confidence physical wall pixels.
    evolved &= ~((wall_probability >= 0.92) & ~dilation(original, disk(1)))
    # Closing repairs one-pixel notches; avoid opening here because it can erase
    # the narrow bridge that is diagnostically important for touching doublets.
    evolved = closing(evolved, disk(1))
    evolved = binary_fill_holes(evolved)
    evolved = remove_small_objects(evolved, max_size=2)
    evolved = _selected_component(
        evolved,
        seed_weights,
        retain_nearby=retain_nearby,
        nearby_distance=nearby_distance,
    )
    evolved = binary_fill_holes(evolved)

    original_area = int(original.sum())
    evolved_area = int(evolved.sum())
    area_ratio = float(evolved_area / max(original_area, 1))
    overlap = _mask_iou(original, evolved)
    center = tuple(int(round((axis - 1) / 2)) for axis in original.shape)
    lost_center_seed = bool(original[center] and not evolved[center])
    status = "refined"
    if not evolved.any():
        status = "fallback_empty"
    elif area_ratio < float(minimum_area_ratio):
        status = "fallback_area_shrink"
    elif overlap < float(minimum_iou):
        status = "fallback_low_iou"
    elif lost_center_seed:
        status = "fallback_seed_lost"
    if diagnostics is not None:
        diagnostics.update(status=status, area_ratio=area_ratio, iou=overlap)
    return original if status.startswith("fallback_") else evolved


def lightweight_refine_instance_mask(mask: np.ndarray) -> np.ndarray:
    """Cheap contour cleanup for confident non-cell objects.

    Debris existence matters to the plate conclusion, but an expensive active
    contour is unnecessary unless the candidate could be a cell.
    """

    output = remove_small_objects(mask.astype(bool), max_size=2)
    output = closing(output, disk(1))
    output = remove_small_holes(output, max_size=9)
    return _selected_component(output)


def _path_signature(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.exists():
        return {"path": str(value.resolve()), "missing": True}
    stat = value.stat()
    return {
        "path": str(value.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _inference_fingerprint(
    config: dict[str, Any],
    predictions_path: Path,
    checkpoint_path: str | Path,
) -> str:
    database = artifact_path(config, "annotations", "annotations.db")
    manual_overrides = config.get("review_queue", {}).get(
        "apply_manual_point_overrides", True
    )
    payload = {
        "algorithm": "v2-instance-contour-guard-20260812",
        "predictions": _path_signature(predictions_path),
        "checkpoint": _path_signature(checkpoint_path),
        "annotations": _path_signature(database) if manual_overrides else {"ignored": True},
        "settings": config.get("v2_inference", {}),
        "manual_overrides": manual_overrides,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contour(mask: np.ndarray, origin_x: int, origin_y: int) -> str:
    # Smooth in probability space, extract at 4x resolution and then apply a
    # light periodic coordinate filter. This removes pixel stairs while keeping
    # the contour within roughly one source pixel of the refined mask.
    surface = zoom(gaussian_filter(mask.astype(np.float32), sigma=0.65), 4, order=3)
    contours = find_contours(surface, 0.5)
    if not contours:
        return "[]"
    # Prefer the enclosing boundary by polygon area.  Internal texture holes
    # can occasionally have more samples than a compact outer boundary after
    # interpolation, so point count alone is not a safe selector.
    def polygon_area(points: np.ndarray) -> float:
        y, x = points[:, 0], points[:, 1]
        return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)

    points = max(contours, key=polygon_area) / 4.0
    points[:, 0] = gaussian_filter1d(points[:, 0], sigma=1.1, mode="wrap")
    points[:, 1] = gaussian_filter1d(points[:, 1], sigma=1.1, mode="wrap")
    stride = max(1, int(np.ceil(len(points) / 120)))
    output = [[round(float(x + origin_x), 2), round(float(y + origin_y), 2)] for y, x in points[::stride]]
    return json.dumps(output, separators=(",", ":"))


def _global_overlap(first: pd.Series, second: pd.Series, size: int) -> tuple[float, float]:
    first_mask = decode_rle(str(first.v2_mask_rle), size)
    second_mask = decode_rle(str(second.v2_mask_rle), size)
    ax0, ay0 = int(first.v2_mask_origin_x), int(first.v2_mask_origin_y)
    bx0, by0 = int(second.v2_mask_origin_x), int(second.v2_mask_origin_y)
    left, top = max(ax0, bx0), max(ay0, by0)
    right, bottom = min(ax0 + size, bx0 + size), min(ay0 + size, by0 + size)
    if left >= right or top >= bottom:
        return 0.0, 0.0
    a = first_mask[top - ay0 : bottom - ay0, left - ax0 : right - ax0]
    b = second_mask[top - by0 : bottom - by0, left - bx0 : right - bx0]
    intersection = int(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0, 0.0
    area_a, area_b = int(first_mask.sum()), int(second_mask.sum())
    return intersection / max(area_a + area_b - intersection, 1), intersection / max(min(area_a, area_b), 1)


def consolidate_v2_masks(frame: pd.DataFrame, patch_size: int) -> pd.DataFrame:
    output = frame.copy()
    output["v2_is_suppressed"] = False
    output["v2_suppressed_by"] = ""
    output["v2_suppression_reason"] = ""
    output["v2_instance_id"] = ""
    valid = output[output["v2_mask_valid"] & ~output["v2_wall_rejected"]]
    for (_, _), group in valid.groupby(["well", "timepoint"], sort=False):
        priority = group["integrated_label"].map(
            {"cluster_3plus": 3, "touching_doublet": 2, "single": 1, "debris": 1, "uncertain": 0, "invalid": -1}
        ).fillna(-1)
        ordered = group.assign(_group_priority=priority).sort_values(
            ["_group_priority", "v2_instance_confidence", "integrated_confidence"], ascending=False
        )
        keepers: list[int] = []
        for index, row in ordered.iterrows():
            owner = None
            reason = ""
            for kept_index in keepers:
                kept = output.loc[kept_index]
                iou, containment = _global_overlap(row, kept, patch_size)
                group_labels = {str(row.integrated_label), str(kept.integrated_label)}
                threshold = 0.22 if group_labels & {"touching_doublet", "cluster_3plus"} else 0.35
                if iou >= threshold or containment >= 0.65:
                    owner = kept_index
                    reason = f"mask_overlap:iou={iou:.3f},containment={containment:.3f}"
                    break
            if owner is None:
                keepers.append(index)
                output.at[index, "v2_instance_id"] = f"{row.well}-{row.timepoint}-I{len(keepers):03d}"
            else:
                output.at[index, "v2_is_suppressed"] = True
                output.at[index, "v2_suppressed_by"] = str(output.at[owner, "candidate_id"])
                output.at[index, "v2_suppression_reason"] = reason
                output.at[index, "v2_instance_id"] = str(output.at[owner, "v2_instance_id"])
    return output


def _reviewed_positive_ids(config: dict[str, Any], frame: pd.DataFrame | None = None) -> set[str]:
    # External validation/review rounds can explicitly require pure model
    # output.  In that mode historical annotations remain archived for
    # evaluation, but must not alter the segmentation gate or final labels.
    if not config.get("review_queue", {}).get("apply_manual_point_overrides", True):
        return set()
    database = artifact_path(config, "annotations", "annotations.db")
    if not database.exists():
        return set()
    try:
        with sqlite3.connect(database) as connection:
            reviews = pd.read_sql_query(
                "SELECT candidate_id, reviewed_label, updated_at FROM integrated_training_reviews ORDER BY updated_at, integrated_review_id",
                connection,
            ).drop_duplicates("candidate_id", keep="last")
            manual = pd.read_sql_query(
                "SELECT candidate_id, well, timepoint, x_px, y_px, reviewed_label, updated_at FROM quick_missed_objects ORDER BY updated_at, quick_missed_id",
                connection,
            ).drop_duplicates("candidate_id", keep="last")
    except Exception:
        return set()
    protected = set(
        reviews.loc[
            reviews["reviewed_label"].isin(["single", "touching_doublet", "cluster_3plus", "debris"]),
            "candidate_id",
        ].astype(str)
    )
    if frame is not None and not manual.empty:
        manual = manual[manual["reviewed_label"].isin(["single", "touching_doublet", "cluster_3plus", "debris"])]
        for point in manual.itertuples(index=False):
            local = frame[(frame["well"] == point.well) & (frame["timepoint"] == point.timepoint)]
            if local.empty:
                continue
            distances = np.hypot(local["x_px"].astype(float) - float(point.x_px), local["y_px"].astype(float) - float(point.y_px))
            protected.update(local.loc[distances <= 20.0, "candidate_id"].astype(str))
    return protected


def finalize_v2_instances(
    enriched: pd.DataFrame,
    patch_size: int,
    protected_positive_ids: set[str] | None = None,
) -> pd.DataFrame:
    output = enriched.copy()
    protected_positive_ids = protected_positive_ids or set()
    if "v2_original_integrated_label" not in output:
        output["v2_original_integrated_label"] = output["integrated_label"]
    else:
        # Makes re-finalization deterministic after threshold/label-policy changes.
        output["integrated_label"] = output["v2_original_integrated_label"]
    automatic_invalid = (
        output["invalid_probability"].fillna(0).astype(float).ge(0.60)
        & ~output["candidate_id"].astype(str).isin(protected_positive_ids)
    )
    output.loc[automatic_invalid, "integrated_label"] = "invalid"
    output["v2_auto_invalid_probability_rule"] = automatic_invalid
    recoverable = (
        output["v2_mask_valid"]
        & ~output["v2_wall_rejected"]
        & ~automatic_invalid
        & output["integrated_label"].isin(["invalid", "uncertain", "unmarked"])
        & (output["v2_instance_confidence"] >= 0.72)
    )
    # A compact instance can have a conservative boundary confidence while
    # the independent morphology and presence heads are unequivocally cell.
    # Keep this rescue narrow so tiered inference cannot revive wall texture.
    strong_cell_recoverable = (
        output["v2_mask_valid"]
        & ~output["v2_wall_rejected"]
        & ~automatic_invalid
        & output["integrated_label"].isin(["invalid", "uncertain", "unmarked"])
        & output["cell_probability"].fillna(0).astype(float).ge(0.90)
        & output["invalid_probability"].fillna(1).astype(float).lt(0.10)
        & output["v2_objectness"].fillna(0).astype(float).ge(0.75)
        & output["v2_instance_confidence"].fillna(0).astype(float).ge(0.35)
    )
    recoverable |= strong_cell_recoverable
    cell_probability = output["cell_probability"].fillna(0).astype(float)
    debris_probability = output["debris_probability"].fillna(0).astype(float)
    invalid_probability = output["invalid_probability"].fillna(0).astype(float)
    # A multiplicity head answers only how many cells an object would contain
    # *if it is a cell*.  It must not turn a debris-dominant object into a
    # single/doublet merely because the instance mask is recoverable.
    cell_rescue = (
        recoverable
        & cell_probability.ge(0.28)
        & cell_probability.ge(debris_probability)
        & cell_probability.ge(invalid_probability)
    )
    debris_rescue = (
        recoverable
        & ~cell_rescue
        & debris_probability.ge(cell_probability)
        & debris_probability.ge(invalid_probability)
    )
    if "predicted_multiplicity" in output:
        rescued = output.loc[cell_rescue, "predicted_multiplicity"].fillna("single")
        rescued = rescued.where(rescued.isin(["single", "touching_doublet", "cluster_3plus"]), "single")
        output.loc[cell_rescue, "integrated_label"] = rescued
    else:
        output.loc[cell_rescue, "integrated_label"] = "single"
    output.loc[debris_rescue, "integrated_label"] = "debris"
    output.loc[recoverable & ~cell_rescue & ~debris_rescue, "integrated_label"] = "uncertain"
    # "Uncertain" is reserved for the biologically relevant cell-vs-noncell
    # boundary.  Once cell probability is low, debris-vs-invalid ambiguity is
    # resolved deterministically and should not consume human review time.
    low_cell_noncell = (
        output["integrated_label"].isin(["uncertain", "unmarked"])
        & output["cell_probability"].fillna(0).astype(float).lt(
            LOW_CELL_NONCELL_RESOLUTION_THRESHOLD
        )
    )
    low_cell_debris = low_cell_noncell & (
        output["debris_probability"].fillna(0).astype(float)
        >= output["invalid_probability"].fillna(0).astype(float)
    )
    output.loc[low_cell_debris, "integrated_label"] = "debris"
    output.loc[low_cell_noncell & ~low_cell_debris, "integrated_label"] = "invalid"
    output["v2_low_cell_noncell_resolved"] = low_cell_noncell
    output["v2_noncell_resolution_label"] = np.where(
        low_cell_debris,
        "debris",
        np.where(low_cell_noncell, "invalid", ""),
    )
    output["v2_rescued_from_invalid"] = recoverable & (output["v2_original_integrated_label"] == "invalid")
    output = consolidate_v2_masks(output, patch_size)
    # Keep instance identity, temporal eligibility, review visibility, and
    # final counting as separate states.  A valid low-confidence object must
    # remain available to the temporal model even when it is not yet suitable
    # for formal counting or review.
    output["v2_is_unique_instance"] = (
        output["v2_mask_valid"]
        & ~output["v2_wall_rejected"]
        & ~output["v2_is_suppressed"]
    )
    output["v2_is_temporal_candidate"] = (
        output["v2_is_unique_instance"]
        & ~automatic_invalid
    )
    output["v2_is_reviewable_instance"] = (
        output["v2_is_unique_instance"]
        & output["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris", "uncertain"]
        )
    )
    output["v2_is_counting_instance"] = (
        output["v2_is_unique_instance"]
        & output["integrated_label"].isin(
            ["single", "touching_doublet", "cluster_3plus", "debris"]
        )
    )
    return output


def refinalize_v2_file(config: dict[str, Any], patch_size: int = 96) -> Path:
    output_path = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    source_frame = pd.read_csv(output_path, low_memory=False)
    frame = finalize_v2_instances(
        source_frame,
        patch_size,
        _reviewed_positive_ids(config, source_frame),
    )
    summary_path = output_path.with_suffix(".json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary.get("checkpoint", ""))
        if checkpoint.exists():
            frame["integrated_round_id"] = "v2-round-" + datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    frame.to_csv(output_path, index=False)
    return output_path


def infer_v2_instances(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    predictions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    pre_temporal_output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> Path:
    """Run V2 inference, optionally writing to an isolated versioned output.

    The default paths intentionally remain the historical ``latest_*`` paths
    so existing production callers keep their behavior.  Review workflows can
    pass explicit paths to create a reproducible batch without overwriting the
    active prediction artifacts.
    """

    predictions_path = Path(predictions_path) if predictions_path is not None else artifact_path(
        config, "predictions", "latest_integrated_predictions.csv"
    )
    output = Path(output_path) if output_path is not None else artifact_path(
        config, "predictions", "latest_v2_predictions.csv"
    )
    pre_temporal_output = (
        Path(pre_temporal_output_path)
        if pre_temporal_output_path is not None
        else output.with_name("latest_v2_pre_temporal_predictions.csv")
    )
    summary_path = Path(summary_path) if summary_path is not None else output.with_suffix(".json")
    for destination in (output, pre_temporal_output, summary_path):
        destination.parent.mkdir(parents=True, exist_ok=True)
    inference_settings = config.get("v2_inference", {})
    fingerprint = _inference_fingerprint(config, predictions_path, checkpoint_path)
    if bool(inference_settings.get("reuse_unchanged_stage", True)) and pre_temporal_output.exists() and summary_path.exists():
        try:
            cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_summary = {}
        if cached_summary.get("inference_fingerprint") == fingerprint:
            shutil.copyfile(pre_temporal_output, output)
            return output

    started = time.perf_counter()
    frame = pd.read_csv(predictions_path, low_memory=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    size = int(checkpoint["patch_size_px"])
    sigma = float(checkpoint["seed_sigma_px"])
    threshold = float(checkpoint.get("threshold", 0.5))
    wall_threshold = float(checkpoint.get("wall_threshold", 0.5))
    presence_threshold = float(checkpoint.get("presence_threshold", 0.55))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeededInstanceUNet(int(checkpoint["base_channels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    protected_positive_ids = _reviewed_positive_ids(config, frame)
    hard_invalid_threshold = float(inference_settings.get("hard_invalid_skip_threshold", 0.60))
    precise_cell_threshold = float(inference_settings.get("precise_contour_cell_probability", 0.12))
    refinement_minimum_area_ratio = float(
        inference_settings.get("refinement_minimum_area_ratio", 0.80)
    )
    refinement_minimum_iou = float(
        inference_settings.get("refinement_minimum_iou", 0.70)
    )
    group_component_gap_px = int(
        inference_settings.get("group_component_gap_px", 3)
    )
    precise_labels = {
        str(value)
        for value in inference_settings.get(
            "precise_contour_labels",
            ["single", "touching_doublet", "cluster_3plus", "uncertain"],
        )
    }
    precise_sources = {
        str(value)
        for value in inference_settings.get(
            "precise_contour_sources",
            ["manual_cell_anchor", "manual_annotation_anchor", "wall_cell_rescue_peak", "wall_residual_peak"],
        )
    }
    skipped_hard_invalid = 0
    precise_refinement_count = 0
    lightweight_refinement_count = 0

    columns: dict[str, list[Any]] = {
        "v2_mask_valid": [], "v2_mask_rle": [], "v2_mask_origin_x": [], "v2_mask_origin_y": [],
        "v2_contour_json": [], "v2_instance_area_px": [], "v2_instance_diameter_px": [],
        "v2_instance_confidence": [], "v2_objectness": [], "v2_wall_overlap": [], "v2_wall_rejected": [],
        "v2_refinement_status": [], "v2_refinement_area_ratio": [], "v2_refinement_iou": [],
    }
    seed = _seed_heatmap(size, sigma)
    for raw_path, group in frame.groupby("raw_image_path", sort=False):
        with Image.open(raw_path) as image:
            full = np.asarray(image.convert("L"), dtype=np.uint8)
        batch_values: list[np.ndarray] = []
        origins: list[tuple[int, int]] = []
        candidate_presence_thresholds: list[float] = []
        precise_refinement: list[bool] = []
        batch_positions: list[int] = []
        group_results: list[dict[str, Any] | None] = [None] * len(group)
        for position, row in enumerate(group.itertuples(index=False)):
            candidate_id = str(row.candidate_id)
            hard_invalid = (
                float(row.invalid_probability) >= hard_invalid_threshold
                and candidate_id not in protected_positive_ids
            )
            if hard_invalid:
                skipped_hard_invalid += 1
                group_results[position] = {
                    "v2_mask_valid": False,
                    "v2_mask_rle": "[]",
                    "v2_mask_origin_x": int(round(row.x_px)) - size // 2,
                    "v2_mask_origin_y": int(round(row.y_px)) - size // 2,
                    "v2_contour_json": "[]",
                    "v2_instance_area_px": 0,
                    "v2_instance_diameter_px": 0.0,
                    "v2_instance_confidence": 0.0,
                    "v2_objectness": 0.0,
                    "v2_wall_overlap": 0.0,
                    "v2_wall_rejected": True,
                    "v2_refinement_status": "skipped_hard_invalid",
                    "v2_refinement_area_ratio": 0.0,
                    "v2_refinement_iou": 0.0,
                }
                continue
            raw = _crop(full, row.x_px, row.y_px, size, int(np.median(full)))
            inner = getattr(row, "detected_wall_inner_fraction", np.nan)
            if pd.isna(inner):
                inner = config.get("candidate_filter", {}).get("hard_wall_exclusion_fraction", 0.44)
            wall = _wall_prior(full.shape, row.x_px, row.y_px, size, float(inner))
            batch_values.append(np.stack([_normalize(raw), seed, wall]))
            origins.append((int(round(row.x_px)) - size // 2, int(round(row.y_px)) - size // 2))
            # Presence is a proposal-quality gate, not a second morphology
            # classifier.  Near-wall cells can have a low global presence mean
            # even when the object head contains a clean, compact mask.  Keep
            # the strict gate for background proposals, but allow a calibrated
            # low gate when independent morphology or human evidence already
            # says that a real object is present.  The >=0.60 invalid rule is
            # deliberately checked first, so texture negatives are not rescued.
            morphology_positive = (
                str(row.integrated_label)
                in {"single", "touching_doublet", "cluster_3plus", "debris", "uncertain"}
                and float(row.invalid_probability) < 0.60
            )
            has_positive_evidence = candidate_id in protected_positive_ids or morphology_positive
            candidate_presence_thresholds.append(min(presence_threshold, 0.08) if has_positive_evidence else presence_threshold)
            source = str(getattr(row, "candidate_source", ""))
            precise_refinement.append(
                candidate_id in protected_positive_ids
                or float(row.cell_probability) >= precise_cell_threshold
                or str(row.integrated_label) in precise_labels
                or source in precise_sources
            )
            batch_positions.append(position)
        for start in range(0, len(batch_values), 128):
            tensor = torch.from_numpy(np.asarray(batch_values[start : start + 128], dtype=np.float32)).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probabilities = torch.sigmoid(model(tensor)).float().cpu().numpy()
            for offset, probability in enumerate(probabilities):
                batch_index = start + offset
                object_probability, wall_probability, presence_probability = probability
                objectness = float(presence_probability.mean())
                local_presence_threshold = candidate_presence_thresholds[batch_index]
                position = batch_positions[batch_index]
                row = group.iloc[position]
                retain_nearby = str(row.integrated_label) in {
                    "touching_doublet", "cluster_3plus"
                }
                binary = object_probability >= threshold
                binary &= ~((wall_probability >= max(wall_threshold, 0.82)) & (object_probability < 0.72))
                mask = _selected_component(
                    binary,
                    seed,
                    retain_nearby=retain_nearby,
                    nearby_distance=group_component_gap_px,
                )
                refinement_diagnostics: dict[str, Any] = {
                    "status": "not_run",
                    "area_ratio": 1.0 if mask.any() else 0.0,
                    "iou": 1.0 if mask.any() else 0.0,
                }
                if objectness >= local_presence_threshold and mask.any():
                    if precise_refinement[batch_index]:
                        precise_refinement_count += 1
                        mask = refine_instance_mask(
                            mask,
                            batch_values[batch_index][0],
                            wall_probability,
                            seed,
                            retain_nearby=retain_nearby,
                            nearby_distance=group_component_gap_px,
                            minimum_area_ratio=refinement_minimum_area_ratio,
                            minimum_iou=refinement_minimum_iou,
                            diagnostics=refinement_diagnostics,
                        )
                    else:
                        lightweight_refinement_count += 1
                        mask = lightweight_refine_instance_mask(mask)
                        refinement_diagnostics["status"] = "lightweight"
                regions = regionprops(mask.astype(np.uint8))
                origin_x, origin_y = origins[batch_index]
                valid = bool(objectness >= local_presence_threshold and regions and 3 <= regions[0].area <= 2200)
                if valid:
                    region = regions[0]
                    area = int(region.area)
                    diameter = float(region.equivalent_diameter_area)
                    confidence = float(object_probability[mask].mean())
                    overlap = float((wall_probability[mask] >= wall_threshold).mean())
                    elongated = bool(region.eccentricity > 0.97 or region.solidity < 0.25)
                    wall_rejected = bool(overlap > 0.86 and elongated and confidence < 0.88)
                else:
                    area, diameter, confidence, overlap, wall_rejected = 0, 0.0, 0.0, 0.0, True
                group_results[batch_positions[batch_index]] = {
                    "v2_mask_valid": valid,
                    "v2_mask_rle": _rle(mask) if valid else "[]",
                    "v2_mask_origin_x": origin_x,
                    "v2_mask_origin_y": origin_y,
                    "v2_contour_json": _contour(mask, origin_x, origin_y) if valid else "[]",
                    "v2_instance_area_px": area,
                    "v2_instance_diameter_px": diameter,
                    "v2_instance_confidence": confidence,
                    "v2_objectness": objectness,
                    "v2_wall_overlap": overlap,
                    "v2_wall_rejected": wall_rejected,
                    "v2_refinement_status": str(refinement_diagnostics["status"]),
                    "v2_refinement_area_ratio": float(refinement_diagnostics["area_ratio"]),
                    "v2_refinement_iou": float(refinement_diagnostics["iou"]),
                }
        for result in group_results:
            if result is None:
                raise RuntimeError(f"V2 inference did not produce a result for {raw_path}")
            for name in columns:
                columns[name].append(result[name])
    # groupby processing changes row order; align through an explicit grouped index list
    ordered_indices = [index for _, group in frame.groupby("raw_image_path", sort=False) for index in group.index]
    enriched = frame.loc[ordered_indices].copy()
    for name, values in columns.items():
        enriched[name] = values
    enriched = enriched.sort_index()
    enriched = finalize_v2_instances(enriched, size, _reviewed_positive_ids(config, enriched))
    enriched["integrated_round_id"] = "v2-round-" + datetime.fromtimestamp(Path(checkpoint_path).stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    enriched.to_csv(pre_temporal_output, index=False)
    enriched.to_csv(output, index=False)
    summary = {
        "algorithm_version": "v2-tiered",
        "source_predictions": str(predictions_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "inference_fingerprint": fingerprint,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "candidate_count": int(len(enriched)),
        "valid_instance_count": int(enriched["v2_mask_valid"].sum()),
        "wall_rejected_count": int(enriched["v2_wall_rejected"].sum()),
        "duplicate_or_covered_count": int(enriched["v2_is_suppressed"].sum()),
        "unique_instance_count": int(enriched["v2_is_unique_instance"].sum()),
        "temporal_candidate_count": int(enriched["v2_is_temporal_candidate"].sum()),
        "reviewable_instance_count": int(enriched["v2_is_reviewable_instance"].sum()),
        "counting_instance_count": int(enriched["v2_is_counting_instance"].sum()),
        "auto_invalid_probability_count": int(enriched["v2_auto_invalid_probability_rule"].sum()),
        "hard_invalid_segmentation_skipped": skipped_hard_invalid,
        "precise_contour_refinement_count": precise_refinement_count,
        "lightweight_contour_refinement_count": lightweight_refinement_count,
        "contour_refinement_status_counts": {
            str(key): int(value)
            for key, value in enriched["v2_refinement_status"].value_counts().items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
