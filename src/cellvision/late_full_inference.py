from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import artifact_path
from .dense_candidates import augment_candidates_with_dense_raw_proposals
from .model_inference import (
    predict_multiplicity_checkpoint,
    predict_teaching_checkpoint,
)
from .multiplicity import (
    ensure_integrated_review_table,
    ensure_multiplicity_table,
    generate_integrated_training_round,
)
from .pseudo_labels import (
    _add_background_anisotropy,
    _add_temporal_support,
    _add_wall_neighbors,
    _assign_pseudo_labels,
    _component_rows,
)
from .teaching import (
    ensure_teaching_features,
    ensure_teaching_table,
    generate_auto_annotation_round,
)
from .v2_instance_inference import infer_v2_instances


LATE_TO_SURROGATE = {"T3": "T1", "T4": "T2"}
CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
UNIT_COUNT = {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}


def _shared_checkpoint(config: dict[str, Any], *parts: str) -> Path:
    shared_root = Path(
        config.get("late_growth", {}).get(
            "shared_model_root",
            Path(__file__).resolve().parents[2] / "artifacts",
        )
    )
    checkpoint = shared_root.joinpath(*parts)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Shared late-growth checkpoint is missing: {checkpoint}")
    return checkpoint


def _late_pipeline_config(config: dict[str, Any]) -> dict[str, Any]:
    late_config = deepcopy(config)
    late_root = Path(config["paths"]["artifact_root"]) / "late_full_pipeline"
    late_config["paths"]["artifact_root"] = str(late_root)
    late_config.setdefault("review_queue", {})["excluded_wells"] = []
    late_config["review_queue"]["apply_manual_point_overrides"] = False
    late_config.setdefault("dense_detection", {})["rebuild_timepoints"] = [
        "T1",
        "T2",
    ]
    late_config["dense_detection"]["include_manual_anchors"] = False
    return late_config


def _write_late_manifest(
    config: dict[str, Any],
    late_config: dict[str, Any],
    selected_wells: set[str] | None,
) -> pd.DataFrame:
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    selected = images[
        images["timepoint"].astype(str).isin(LATE_TO_SURROGATE)
        & images["decode_status"].astype(str).eq("ok")
    ].copy()
    excluded = {
        str(value).upper()
        for value in config.get("review_queue", {}).get("excluded_wells", [])
    }
    selected = selected[
        ~selected["well"].astype(str).str.upper().isin(excluded)
    ].copy()
    if selected_wells is not None:
        selected = selected[
            selected["well"].astype(str).str.upper().isin(selected_wells)
        ].copy()
    selected["late_timepoint"] = selected["timepoint"].astype(str)
    selected["timepoint"] = selected["late_timepoint"].map(LATE_TO_SURROGATE)
    selected.to_csv(
        artifact_path(late_config, "manifests", "images.csv"),
        index=False,
        encoding="utf-8",
    )
    return selected


def _write_base_candidates(
    late_config: dict[str, Any], images: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image in images.itertuples(index=False):
        for row in _component_rows(
            image.cf_image_path,
            str(image.well),
            str(image.timepoint),
            {"align_shift_x_px": 0.0, "align_shift_y_px": 0.0},
        ):
            row["raw_image_path"] = image.raw_image_path
            row["cf_image_path"] = image.cf_image_path
            rows.append(row)
    if not rows:
        raise RuntimeError("No T3/T4 CF or raw-image candidates were available.")
    candidates = _assign_pseudo_labels(
        _add_background_anisotropy(
            _add_temporal_support(_add_wall_neighbors(pd.DataFrame(rows)))
        )
    )
    candidates["candidate_source"] = "cf_component"
    candidates["dense_response"] = float("nan")
    candidates["wall_rescue_blobness"] = 0.0
    candidates.to_csv(
        artifact_path(
            late_config, "pseudo_labels", "morphology_candidates.csv"
        ),
        index=False,
        encoding="utf-8",
    )
    return candidates


def _boolean_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    return values.fillna(default).astype(str).str.lower().isin({"true", "1", "yes"})


def _dense_late_group_mask(
    frame: pd.DataFrame,
    *,
    minimum_instances: int,
    minimum_units: int,
) -> pd.Series:
    """Select late images whose growth conclusion no longer needs fine masks.

    Integrated inference has already made candidates mutually exclusive.  If
    one late image contains far more independent cell instances than the
    biological growth threshold, running edge-snapped V2 masks for every cell
    cannot change the well-level conclusion and is prohibitively expensive for
    confluent late images.
    """

    result = pd.Series(False, index=frame.index, dtype=bool)
    if frame.empty:
        return result
    counting = _boolean_series(frame, "is_counting_instance", True)
    labels = frame["integrated_label"].astype(str)
    cell = labels.isin(CELL_LABELS) & counting
    group_columns = ["well", "timepoint"]
    metrics = (
        frame.loc[cell, group_columns]
        .assign(
            instance_count=1,
            unit_count=labels.loc[cell].map(UNIT_COUNT).fillna(0).astype(int),
        )
        .groupby(group_columns, sort=False)[["instance_count", "unit_count"]]
        .sum()
    )
    dense_keys = {
        tuple(index)
        for index, row in metrics.iterrows()
        if int(row.instance_count) >= minimum_instances
        or int(row.unit_count) >= minimum_units
    }
    if not dense_keys:
        return result
    return pd.Series(
        [
            (str(well), str(timepoint)) in dense_keys
            for well, timepoint in zip(frame["well"], frame["timepoint"])
        ],
        index=frame.index,
        dtype=bool,
    )


def _dense_v2_passthrough(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise V2-compatible columns without unnecessary fine contours."""

    result = frame.copy()
    counting = _boolean_series(result, "is_counting_instance", True)
    labels = result["integrated_label"].astype(str)
    area = result.get("area_px", pd.Series(0.0, index=result.index)).fillna(0).astype(float)
    result["v2_mask_valid"] = False
    result["v2_mask_rle"] = "[]"
    result["v2_mask_origin_x"] = 0
    result["v2_mask_origin_y"] = 0
    result["v2_contour_json"] = "[]"
    result["v2_instance_area_px"] = area.astype(int)
    result["v2_instance_diameter_px"] = np.sqrt(np.maximum(area, 0.0) * 4.0 / np.pi)
    result["v2_instance_confidence"] = result.get(
        "integrated_confidence", pd.Series(0.0, index=result.index)
    ).fillna(0).astype(float)
    result["v2_objectness"] = result["v2_instance_confidence"]
    result["v2_wall_overlap"] = 0.0
    result["v2_wall_rejected"] = labels.eq("invalid")
    result["v2_original_integrated_label"] = labels
    result["v2_auto_invalid_probability_rule"] = labels.eq("invalid")
    result["v2_low_cell_noncell_resolved"] = False
    result["v2_noncell_resolution_label"] = ""
    result["v2_rescued_from_invalid"] = False
    result["v2_is_suppressed"] = ~counting
    duplicate_owner = result.get(
        "duplicate_of_candidate_id", pd.Series("", index=result.index)
    ).fillna("").astype(str)
    hierarchy_owner = result.get(
        "suppressed_by_candidate_id", pd.Series("", index=result.index)
    ).fillna("").astype(str)
    result["v2_suppressed_by"] = duplicate_owner.where(
        duplicate_owner.ne(""), hierarchy_owner
    )
    result["v2_suppression_reason"] = "late_dense_integrated_passthrough"
    result["v2_instance_id"] = result["candidate_id"].astype(str)
    result["v2_is_unique_instance"] = counting
    result["v2_is_temporal_candidate"] = False
    result["v2_is_reviewable_instance"] = counting & labels.isin(
        CELL_LABELS | {"debris", "uncertain"}
    )
    result["v2_is_counting_instance"] = counting
    result["v2_processing_mode"] = "dense_integrated_passthrough"
    result["integrated_round_id"] = "v2-dense-passthrough-" + datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    return result


def _infer_sparse_v2_with_dense_passthrough(
    config: dict[str, Any], late_config: dict[str, Any]
) -> Path:
    predictions = artifact_path(
        late_config, "predictions", "latest_integrated_predictions.csv"
    )
    frame = pd.read_csv(predictions, low_memory=False)
    settings = config.get("late_growth", {})
    dense_mask = _dense_late_group_mask(
        frame,
        minimum_instances=int(
            settings.get("precise_v2_max_counting_instances", 96)
        ),
        minimum_units=int(settings.get("precise_v2_max_cell_units", 160)),
    )
    dense = _dense_v2_passthrough(frame[dense_mask])
    sparse = frame[~dense_mask].copy()
    print(
        "late V2 workload: "
        f"precise={len(sparse)} candidates, "
        f"dense_passthrough={len(dense)} candidates, "
        f"dense_images={frame.loc[dense_mask, ['well', 'timepoint']].drop_duplicates().shape[0]}",
        flush=True,
    )
    if sparse.empty:
        precise = sparse
    else:
        sparse_config = deepcopy(late_config)
        sparse_root = Path(late_config["paths"]["artifact_root"]) / "v2_sparse"
        sparse_config["paths"]["artifact_root"] = str(sparse_root)
        sparse.to_csv(
            artifact_path(
                sparse_config, "predictions", "latest_integrated_predictions.csv"
            ),
            index=False,
            encoding="utf-8",
        )
        sparse_source = infer_v2_instances(
            sparse_config,
            _shared_checkpoint(
                config, "v2", "models", "latest_instance_segmenter.pt"
            ),
        )
        precise = pd.read_csv(sparse_source, low_memory=False)
        precise["v2_processing_mode"] = "precise_instance_mask"
    result = pd.concat([precise, dense], ignore_index=True, sort=False)
    result = result.sort_values(["well", "timepoint", "candidate_id"])
    output = artifact_path(
        late_config, "predictions", "latest_v2_predictions.csv"
    )
    result.to_csv(output, index=False, encoding="utf-8")
    summary = {
        "algorithm_version": "late-v2-dense-short-circuit-v1",
        "candidate_count": int(len(result)),
        "precise_candidate_count": int(len(precise)),
        "dense_passthrough_candidate_count": int(len(dense)),
        "dense_image_count": int(
            frame.loc[dense_mask, ["well", "timepoint"]].drop_duplicates().shape[0]
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def infer_late_full_instances(
    config: dict[str, Any],
    selected_wells: set[str] | None = None,
    *,
    reuse_existing_integrated: bool = False,
) -> Path:
    """Run the mature T0-T2 detection stack on T3/T4 images.

    T3 and T4 are represented internally as T1 and T2 so the existing dense
    proposal, morphology, multiplicity, and V2 instance code can be reused
    without mixing late predictions into the audited T0-T2 artifacts.
    """

    late_config = _late_pipeline_config(config)
    images = _write_late_manifest(config, late_config, selected_wells)
    integrated_cache = artifact_path(
        late_config, "predictions", "latest_integrated_predictions.csv"
    )
    if not reuse_existing_integrated:
        _write_base_candidates(late_config, images)

    database = artifact_path(late_config, "annotations", "annotations.db")
    ensure_teaching_table(database)
    ensure_multiplicity_table(database)
    ensure_integrated_review_table(database)

    if reuse_existing_integrated:
        if not integrated_cache.exists():
            raise FileNotFoundError(
                f"Late integrated cache is unavailable: {integrated_cache}"
            )
        cached = pd.read_csv(
            integrated_cache,
            usecols=["well", "timepoint", "raw_image_path"],
            low_memory=False,
        )
        expected_paths = set(images["raw_image_path"].astype(str))
        cached_paths = set(cached["raw_image_path"].astype(str))
        if cached_paths != expected_paths:
            raise RuntimeError(
                "Late integrated cache does not match the requested image set."
            )
        print(
            f"late full pipeline: reusing integrated cache for {len(images)} images",
            flush=True,
        )
    else:
        print(
            f"late full pipeline: {len(images)} images; generating T0-T2 dense proposals",
            flush=True,
        )
        augment_candidates_with_dense_raw_proposals(late_config, database)
        ensure_teaching_features(late_config)
        predict_teaching_checkpoint(
            late_config,
            _shared_checkpoint(config, "models", "teaching_classifier.pt"),
        )
        generate_auto_annotation_round(late_config, database)
        predict_multiplicity_checkpoint(
            late_config,
            _shared_checkpoint(config, "models", "multiplicity_classifier.pt"),
        )
        generate_integrated_training_round(late_config, database)
    source = _infer_sparse_v2_with_dense_passthrough(config, late_config)

    result = pd.read_csv(source, low_memory=False)
    result["surrogate_timepoint"] = result["timepoint"].astype(str)
    late_lookup = {
        (str(row.well), str(row.timepoint)): str(row.late_timepoint)
        for row in images.itertuples(index=False)
    }
    result["timepoint"] = [
        late_lookup.get((str(well), str(timepoint)))
        for well, timepoint in zip(
            result["well"], result["surrogate_timepoint"]
        )
    ]
    result = result[result["timepoint"].notna()].copy()
    output = artifact_path(
        config, "predictions", "latest_late_full_v2_predictions.csv"
    )
    if selected_wells is not None and output.exists():
        previous = pd.read_csv(output, low_memory=False)
        previous = previous[
            ~previous["well"].astype(str).str.upper().isin(selected_wells)
        ]
        result = pd.concat([previous, result], ignore_index=True, sort=False)
    result = result.sort_values(["well", "timepoint", "candidate_id"])
    result.to_csv(output, index=False, encoding="utf-8")
    return output
