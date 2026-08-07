from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .config import artifact_path
from .pseudo_labels import _background_anisotropy


def rank_uncertain(frame: pd.DataFrame, probability_columns: list[str]) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["uncertainty"] = 1.0 - ranked[probability_columns].max(axis=1)
    return ranked.sort_values(["uncertainty", "well"], ascending=[False, True])


def filter_review_candidates(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    excluded = {str(well).upper() for well in settings.get("excluded_wells", [])}
    minimum_fraction = float(settings.get("minimum_foreground_fraction", 0.001))
    minimum_confidence = float(settings.get("minimum_confidence", 0.65))
    filtered = frame[
        ~frame["well"].str.upper().isin(excluded)
        & (frame["foreground_fraction"].astype(float) >= minimum_fraction)
        & (frame["confidence"].astype(float) >= minimum_confidence)
    ].copy()
    preferred = settings.get("prefer_candidate_sources", ["cf_component", "instrument_csv"])
    priority = {source: index for index, source in enumerate(preferred)}
    filtered["source_priority"] = filtered["candidate_source"].map(priority).fillna(len(priority))
    return filtered.sort_values(
        ["source_priority", "confidence", "foreground_fraction"],
        ascending=[True, False, False],
    ).drop(columns=["source_priority"])


def latest_prediction_file(config: dict[str, Any]) -> Path | None:
    prediction_root = Path(config["paths"]["artifact_root"]) / "predictions"
    files = sorted(
        prediction_root.glob("weak-seg-*/candidate_patch_predictions.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def completed_lineage_targets(config: dict[str, Any]) -> set[str]:
    database = (
        Path(config["paths"]["artifact_root"])
        / "annotations"
        / "annotations.db"
    )
    if not database.exists():
        return set()
    try:
        with sqlite3.connect(database) as connection:
            return {
                str(row[0])
                for row in connection.execute(
                    "SELECT canonical_target_id FROM lineage_reviews"
                ).fetchall()
            }
    except sqlite3.OperationalError:
        return set()


def model_lineage_candidates(
    config: dict[str, Any], *, include_completed: bool = False
) -> pd.DataFrame | None:
    """Build the temporal-review queue from the latest taught 3-class model."""
    proposal_source = (
        Path(config["paths"]["artifact_root"])
        / "predictions"
        / "latest_tracking_proposals.csv"
    )
    integrated_source = (
        Path(config["paths"]["artifact_root"])
        / "predictions"
        / "latest_integrated_predictions.csv"
    )
    if proposal_source.exists() and integrated_source.exists():
        proposals = pd.read_csv(proposal_source)
        integrated = pd.read_csv(
            integrated_source,
            usecols=["integrated_round_id"],
            nrows=1,
        )
        if (
            not proposals.empty
            and not integrated.empty
            and str(proposals.iloc[0]["round_id"])
            == str(integrated.iloc[0]["integrated_round_id"])
        ):
            return proposals

    source = (
        Path(config["paths"]["artifact_root"])
        / "predictions"
        / "latest_auto_annotations.csv"
    )
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    required = {
        "candidate_id",
        "well",
        "timepoint",
        "x_px",
        "y_px",
        "area_px",
        "auto_label",
        "auto_status",
        "cell_probability",
        "debris_probability",
        "invalid_probability",
    }
    if not required.issubset(frame.columns):
        return None

    settings = config.get("review_queue", {})
    excluded = {
        str(well).upper() for well in settings.get("excluded_wells", [])
    }
    minimum_cell_probability = float(
        settings.get("model_cell_probability_min", 0.25)
    )
    minimum_debris_probability = float(
        settings.get("model_debris_probability_min", 0.80)
    )
    eligible = frame[
        (frame["timepoint"].astype(str) == "T0")
        & ~frame["well"].astype(str).str.upper().isin(excluded)
    ].copy()
    cell_probability = eligible["cell_probability"].astype(float)
    selected = eligible[
        (eligible["auto_label"].astype(str) == "cell")
        | (
            (eligible["auto_label"].astype(str) == "debris")
            & (
                eligible["debris_probability"].astype(float)
                >= minimum_debris_probability
            )
        )
        | (
            (eligible["auto_status"].astype(str) == "needs_review")
            & (cell_probability >= minimum_cell_probability)
        )
    ].copy()
    selected["candidate_source"] = selected["auto_label"].map(
        {
            "cell": "trained_model_cell",
            "debris": "trained_model_debris",
        }
    ).fillna("trained_model_uncertain")
    selected["decision_source"] = "latest_three_class_model"
    selected["model_confidence"] = selected.get(
        "confidence", selected["cell_probability"]
    ).astype(float)
    selected["confidence"] = selected["cell_probability"].astype(float)
    selected["foreground_fraction"] = (
        selected["area_px"].astype(float) / 96**2
    ).clip(lower=0.0001, upper=1.0)
    selected["review_priority"] = (
        selected["cell_probability"].astype(float)
        + 0.2 * (selected["auto_label"].astype(str) == "cell").astype(float)
        + 0.1
        * (selected["auto_label"].astype(str) == "debris").astype(float)
        * selected["debris_probability"].astype(float)
    )

    maximum_per_well = int(
        settings.get(
            "model_max_candidates_per_well",
            settings.get("max_candidates_per_well", 8),
        )
    )
    selected = selected.sort_values(
        ["well", "review_priority", "area_px"],
        ascending=[True, False, False],
    )
    selected = selected.groupby("well", group_keys=False).head(maximum_per_well)

    if include_completed:
        completed = completed_lineage_targets(config)
        if completed:
            historical = eligible[
                eligible["candidate_id"].astype(str).isin(completed)
            ].copy()
            if not historical.empty:
                historical["candidate_source"] = np.where(
                    historical["auto_label"].astype(str) == "cell",
                    "trained_model_cell",
                    "reviewed_model_history",
                )
                historical["decision_source"] = "existing_human_review"
                historical["model_confidence"] = historical.get(
                    "confidence", historical["cell_probability"]
                ).astype(float)
                historical["confidence"] = historical[
                    "cell_probability"
                ].astype(float)
                historical["foreground_fraction"] = (
                    historical["area_px"].astype(float) / 96**2
                ).clip(lower=0.0001, upper=1.0)
                historical["review_priority"] = (
                    historical["cell_probability"].astype(float)
                )
                selected = pd.concat(
                    [selected, historical], ignore_index=True
                ).drop_duplicates("candidate_id", keep="first")
            missing_completed = completed - set(
                selected["candidate_id"].astype(str)
            )
            if missing_completed:
                database = (
                    Path(config["paths"]["artifact_root"])
                    / "annotations"
                    / "annotations.db"
                )
                historical_rows: list[dict[str, Any]] = []
                try:
                    with sqlite3.connect(database) as connection:
                        connection.row_factory = sqlite3.Row
                        placeholders = ",".join("?" for _ in missing_completed)
                        reviews = connection.execute(
                            "SELECT canonical_target_id, well, "
                            "timepoint_points_json FROM lineage_reviews "
                            f"WHERE canonical_target_id IN ({placeholders})",
                            tuple(sorted(missing_completed)),
                        ).fetchall()
                    for review in reviews:
                        points = json.loads(
                            review["timepoint_points_json"] or "{}"
                        )
                        t0 = points.get("T0", {})
                        if t0.get("x_px") is None or t0.get("y_px") is None:
                            continue
                        historical_rows.append(
                            {
                                "candidate_id": review[
                                    "canonical_target_id"
                                ],
                                "well": review["well"],
                                "timepoint": "T0",
                                "x_px": t0["x_px"],
                                "y_px": t0["y_px"],
                                "area_px": t0.get("area_px"),
                                "diameter_px": t0.get(
                                    "marker_diameter_px", 8
                                ),
                                "auto_label": "historical",
                                "auto_status": "human_reviewed",
                                "cell_probability": np.nan,
                                "debris_probability": np.nan,
                                "invalid_probability": np.nan,
                                "candidate_source": "human_review_history",
                                "decision_source": "existing_human_review",
                                "model_confidence": np.nan,
                                "confidence": 1.0,
                                "foreground_fraction": 0.001,
                                "review_priority": 0.0,
                            }
                        )
                except (sqlite3.OperationalError, json.JSONDecodeError):
                    historical_rows = []
                if historical_rows:
                    selected = pd.concat(
                        [selected, pd.DataFrame(historical_rows)],
                        ignore_index=True,
                    ).drop_duplicates("candidate_id", keep="first")
    return selected


def morphology_review_candidates(config: dict[str, Any]) -> pd.DataFrame | None:
    source = (
        Path(config["paths"]["artifact_root"])
        / "pseudo_labels"
        / "morphology_candidates.csv"
    )
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    required = {
        "candidate_id",
        "well",
        "timepoint",
        "x_px",
        "y_px",
        "area_px",
        "radial_fraction",
        "pseudo_label",
    }
    if not required.issubset(frame.columns):
        return None

    settings = config.get("review_queue", {})
    excluded = {
        str(well).upper() for well in settings.get("excluded_wells", [])
    }
    selected = frame[
        (frame["timepoint"] == "T0")
        & ~frame["well"].str.upper().isin(excluded)
        & frame["pseudo_label"].isin(["cell", "uncertain"])
    ].copy()
    if selected.empty:
        return selected

    anisotropy_cache_path = artifact_path(
        config, "cache", "review_candidate_anisotropy.csv"
    )
    measured = pd.DataFrame(columns=["candidate_id", "review_anisotropy"])
    if anisotropy_cache_path.exists():
        measured = pd.read_csv(anisotropy_cache_path)
    measured_ids = set(measured.get("candidate_id", pd.Series(dtype=str)).astype(str))
    missing = selected[
        ~selected["candidate_id"].astype(str).isin(measured_ids)
    ].copy()
    new_rows: list[dict[str, Any]] = []
    if not missing.empty and "raw_image_path" in missing.columns:
        groups = list(missing.groupby("raw_image_path", sort=False))
        for group_index, (raw_path, group) in enumerate(groups, start=1):
            with Image.open(raw_path) as image:
                raw = np.asarray(image.convert("L"), dtype=np.uint8)
            for row in group.itertuples(index=False):
                new_rows.append(
                    {
                        "candidate_id": row.candidate_id,
                        "review_anisotropy": _background_anisotropy(
                            raw, row.x_px, row.y_px
                        ),
                    }
                )
            if group_index % 40 == 0 or group_index == len(groups):
                print(
                    f"wall-feature cache: {group_index}/{len(groups)} images",
                    flush=True,
                )
    elif not missing.empty:
        for row in missing.itertuples(index=False):
            new_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "review_anisotropy": float(
                        getattr(row, "background_anisotropy", 1.0)
                    ),
                }
            )
    if new_rows:
        measured = pd.concat(
            [measured, pd.DataFrame(new_rows)], ignore_index=True
        ).drop_duplicates("candidate_id", keep="last")
        measured.to_csv(anisotropy_cache_path, index=False, encoding="utf-8")
    selected = selected.merge(measured, on="candidate_id", how="left")
    anisotropy = selected["review_anisotropy"].fillna(
        selected.get(
            "background_anisotropy", pd.Series(1.0, index=selected.index)
        )
    ).astype(float)

    wall_start = float(
        settings.get(
            "auto_invalid_wall_start_fraction",
            config.get("candidate_filter", {}).get(
                "wall_band_start_fraction", 0.42
            ),
        )
    )
    directional_threshold = float(
        settings.get("auto_invalid_wall_anisotropy", 0.55)
    )
    chain_threshold = float(
        settings.get("auto_invalid_wall_chain_anisotropy", 0.40)
    )
    wall_chain = selected.get(
        "wall_neighbor_count", pd.Series(0, index=selected.index)
    ).astype(float) >= int(
        config.get("candidate_filter", {}).get(
            "wall_chain_min_neighbors", 2
        )
    )
    in_wall_band = selected["radial_fraction"].astype(float) >= wall_start
    invalid_wall = in_wall_band & (
        (anisotropy >= directional_threshold)
        | (wall_chain & (anisotropy >= chain_threshold))
    ) & ~selected.get(
        "candidate_source", pd.Series("", index=selected.index)
    ).astype(str).eq("wall_residual_peak")
    auto_invalid = selected[invalid_wall].copy()
    auto_invalid["auto_decision"] = "invalid"
    auto_invalid["decision_reason"] = "directional_well_wall_structure"
    auto_invalid["review_anisotropy"] = anisotropy[invalid_wall]
    auto_invalid.to_csv(
        artifact_path(config, "annotations", "auto_invalid_wall_candidates.csv"),
        index=False,
        encoding="utf-8",
    )
    selected = selected[~invalid_wall].copy()
    anisotropy = selected["review_anisotropy"].fillna(1.0).astype(float)

    shape_score = (
        selected["circularity"].astype(float).clip(0, 1)
        + selected["solidity"].astype(float).clip(0, 1)
        + selected["extent"].astype(float).clip(0, 1)
    ) / 3
    temporal_score = selected.get(
        "temporal_support", pd.Series(0.0, index=selected.index)
    ).astype(float).clip(0, 2) / 2
    selected["review_priority"] = (
        0.48 * shape_score
        + 0.32 * (1.0 - anisotropy.clip(0, 1))
        + 0.20 * temporal_score
    )
    selected.loc[selected["pseudo_label"] == "cell", "review_priority"] += 0.25
    selected["confidence"] = selected["review_priority"].clip(0, 0.999)
    selected["foreground_fraction"] = (
        selected["area_px"].astype(float) / 96**2
    ).clip(lower=0.0001, upper=1.0)
    selected["candidate_source"] = selected["pseudo_label"].map(
        {
            "cell": "morphology_cell_seed",
            "uncertain": "morphology_uncertain",
        }
    )
    selected["decision_source"] = "full_ql11111_morphology_queue"

    maximum_per_well = int(settings.get("max_candidates_per_well", 8))
    selected = selected.sort_values(
        ["well", "review_priority", "area_px"],
        ascending=[True, False, False],
    )
    return selected.groupby("well", group_keys=False).head(maximum_per_well)


def build_review_queue(
    config: dict[str, Any], *, include_completed: bool = False
) -> pd.DataFrame:
    model_queue = model_lineage_candidates(
        config, include_completed=include_completed
    )
    if model_queue is not None and not model_queue.empty:
        queue = model_queue
    else:
        morphology_queue = morphology_review_candidates(config)
        if morphology_queue is not None and not morphology_queue.empty:
            queue = morphology_queue
        else:
            prediction_file = latest_prediction_file(config)
            if prediction_file is None:
                return pd.DataFrame()
            predictions = pd.read_csv(prediction_file)
            queue = filter_review_candidates(
                predictions, config.get("review_queue", {})
            )
    sequences = pd.read_csv(artifact_path(config, "manifests", "sequences.csv"))
    sequence_lookup = dict(zip(sequences["well"], sequences["sequence_id"]))
    queue["sequence_id"] = queue["well"].map(sequence_lookup)
    queue["plate_id"] = config["experiment"]["plate_id"]
    queue["object_id"] = queue["candidate_id"]
    if "canonical_target_id" not in queue.columns:
        queue["canonical_target_id"] = queue["candidate_id"]
    if not include_completed:
        completed = completed_lineage_targets(config)
        queue = queue[~queue["canonical_target_id"].isin(completed)].copy()
    well_order = {well: index for index, well in enumerate(sequences["well"])}
    queue["_well_order"] = queue["well"].map(well_order).fillna(len(well_order))
    queue = queue.sort_values(
        ["_well_order", "confidence"], ascending=[True, False]
    ).drop(columns=["_well_order"])
    queue = queue.astype(object).where(pd.notna(queue), None)
    destination = artifact_path(config, "annotations", "review_queue.csv")
    queue.to_csv(destination, index=False, encoding="utf-8")
    return queue
