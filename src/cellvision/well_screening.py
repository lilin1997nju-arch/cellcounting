from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import artifact_path
from .hierarchy import suppress_nested_single_candidates


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
CELL_UNITS = {"single": 1, "touching_doublet": 2, "cluster_3plus": 3}
LATE_GROWTH_TIMEPOINTS = ("T3", "T4")
LATE_GROWTH_DECISIONS = {"obvious_growth", "no_growth", "uncertain"}


def _latest_prediction_source(config: dict[str, Any]) -> Path:
    """Return the newest completed prediction table for screening.

    V2 inference writes a richer table than the original integrated pass.  The
    review server already uses the newest completed table, but the well-level
    report used to be pinned to ``latest_integrated_predictions.csv``.  That
    allowed a stale V1 row (for example a wall residual) to affect the T0
    origin count even while the review UI showed the V2 instance result.
    """
    base = artifact_path(config, "predictions", "latest_integrated_predictions.csv")
    candidates = [base]
    for name in (
        "latest_temporally_completed_predictions.csv",
        "latest_v2_predictions.csv",
        "latest_v3_predictions.csv",
    ):
        path = artifact_path(config, "predictions", name)
        if path.exists():
            candidates.append(path)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return base
    return max(existing, key=lambda path: path.stat().st_mtime_ns)


def _boolean_series(values: pd.Series) -> pd.Series:
    """Normalize bool columns after optional manual-row concatenation."""
    def convert(value: Any) -> bool:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    return values.map(convert).astype(bool)


def _visible_v2_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep one visible row per V2 instance and remove rejected wall rows."""
    if "v2_instance_id" not in predictions.columns:
        return predictions
    manual = _boolean_series(
        predictions.get("is_manual_missed", pd.Series(False, index=predictions.index))
    )
    mask_valid = _boolean_series(predictions.get("v2_mask_valid", pd.Series(False, index=predictions.index)))
    wall_rejected = _boolean_series(predictions.get("v2_wall_rejected", pd.Series(False, index=predictions.index)))
    suppressed = _boolean_series(predictions.get("v2_is_suppressed", pd.Series(False, index=predictions.index)))
    visible = predictions[manual | (mask_valid & ~wall_rejected & ~suppressed)].copy()
    instance_id = visible["v2_instance_id"].fillna("").astype(str)
    has_instance = ~instance_id.isin(["", "nan", "None"])
    with_instance = visible[has_instance].copy()
    without_instance = visible[~has_instance].copy()
    if not with_instance.empty:
        with_instance["_reviewed_priority"] = with_instance.get(
            "human_reviewed", pd.Series(False, index=with_instance.index)
        ).astype(bool).astype(int)
        with_instance["_instance_confidence_priority"] = pd.to_numeric(
            with_instance.get(
                "v2_instance_confidence", pd.Series(0.0, index=with_instance.index)
            ), errors="coerce"
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


def ensure_well_screening_review_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS well_screening_reviews (
              well TEXT PRIMARY KEY,
              decision TEXT NOT NULL,
              reviewer TEXT,
              notes TEXT,
              updated_at TEXT NOT NULL
            )
            """
        )


def ensure_well_timepoint_cell_count_review_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS well_timepoint_cell_count_reviews (
              well TEXT NOT NULL,
              timepoint TEXT NOT NULL,
              cell_count INTEGER NOT NULL,
              reviewer TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (well, timepoint)
            )
            """
        )


def ensure_late_growth_review_table(database: str | Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS late_growth_reviews (
              well TEXT NOT NULL,
              timepoint TEXT NOT NULL,
              decision TEXT NOT NULL,
              reviewer TEXT,
              notes TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (well, timepoint)
            )
            """
        )


def save_late_growth_review(
    database: str | Path,
    well: str,
    timepoint: str,
    decision: str,
    reviewer: str,
    notes: str,
    updated_at: str,
) -> None:
    normalized_timepoint = timepoint.upper()
    if normalized_timepoint not in LATE_GROWTH_TIMEPOINTS:
        raise ValueError("Late growth review timepoint must be T3 or T4.")
    if decision not in LATE_GROWTH_DECISIONS:
        raise ValueError("Invalid late growth review decision.")
    ensure_late_growth_review_table(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO late_growth_reviews (
              well, timepoint, decision, reviewer, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(well, timepoint) DO UPDATE SET
              decision=excluded.decision,
              reviewer=excluded.reviewer,
              notes=excluded.notes,
              updated_at=excluded.updated_at
            """,
            (
                well.upper(),
                normalized_timepoint,
                decision,
                reviewer,
                notes,
                updated_at,
            ),
        )


def late_growth_gate(
    available_timepoints: set[str],
    decisions: dict[str, str],
) -> tuple[str, bool]:
    """Return the late-growth status and whether deep search must be skipped.

    Skipping is deliberately conservative: every available late image must
    have an explicit ``no_growth`` decision.  A positive, uncertain, or
    unreviewed late image always keeps the well eligible for deeper analysis.
    """

    available = [
        timepoint
        for timepoint in LATE_GROWTH_TIMEPOINTS
        if timepoint in available_timepoints
    ]
    if not available:
        return "unavailable", False
    values = [decisions.get(timepoint, "pending") for timepoint in available]
    if "obvious_growth" in values:
        return "obvious_growth", False
    if "pending" in values:
        return "pending", False
    if "uncertain" in values:
        return "uncertain", False
    if all(value == "no_growth" for value in values):
        return "no_growth", True
    return "pending", False


def _late_growth_reviews(database: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    ensure_late_growth_review_table(database)
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query("SELECT * FROM late_growth_reviews", connection)
    if frame.empty:
        return {}
    return {
        (str(row.well).upper(), str(row.timepoint).upper()): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _automatic_late_growth(config: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    source = artifact_path(config, "predictions", "latest_late_growth_predictions.csv")
    if not source.exists():
        return {}
    frame = pd.read_csv(source)
    if frame.empty:
        return {}
    return {
        (str(row.well).upper(), str(row.timepoint).upper()): row._asdict()
        for row in frame.itertuples(index=False)
    }


def _latest_reviews(database: str | Path) -> pd.DataFrame:
    try:
        with sqlite3.connect(database) as connection:
            reviews = pd.read_sql_query(
                """
                SELECT integrated_review_id, candidate_id, reviewed_label,
                       updated_at
                FROM integrated_training_reviews
                ORDER BY updated_at, integrated_review_id
                """,
                connection,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame(columns=["candidate_id", "reviewed_label"])
    return reviews.drop_duplicates("candidate_id", keep="last")


def _manual_missed_objects(
    database: str | Path, round_id: str | None = None
) -> pd.DataFrame:
    try:
        with sqlite3.connect(database) as connection:
            manual = pd.read_sql_query(
                """
                SELECT candidate_id, well, timepoint, x_px, y_px,
                       diameter_px, reviewed_label
                FROM quick_missed_objects
                ORDER BY updated_at
                """,
                connection,
            )
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return pd.DataFrame()
    if manual.empty:
        return manual
    manual = manual.drop_duplicates("candidate_id", keep="last")
    manual["integrated_label"] = manual["reviewed_label"].astype(str)
    manual["integrated_confidence"] = 1.0
    manual["screen_label"] = manual["reviewed_label"].astype(str)
    manual["screen_confidence"] = 1.0
    manual["human_reviewed"] = True
    manual["manual_override"] = True
    manual["is_manual_missed"] = True
    manual["v2_suspected_dead_cell"] = False
    manual["area_px"] = np.pi * (
        manual["diameter_px"].astype(float) / 2.0
    ) ** 2
    manual["is_hierarchy_suppressed"] = False
    manual["is_duplicate_suppressed"] = False
    return manual


def _classify_well_status(
    counts: dict[str, int],
    *,
    t0_cell_instances: int,
    t0_has_uncertain: bool,
    confidence: float,
    late_growth_status: str,
) -> tuple[str, bool, bool, bool, bool]:
    """Return status and the origin/growth flags used by the plate report."""

    single_origin = bool(
        counts["T0"] == 1
        and t0_cell_instances == 1
        and not t0_has_uncertain
    )
    multi_origin = bool(counts["T0"] > 1 or t0_cell_instances > 1)
    early_growth = max(counts["T1"], counts["T2"]) >= 2
    late_growth = late_growth_status == "obvious_growth"
    growth_active = bool(single_origin and (early_growth or late_growth))
    # A clear T3/T4 growth event is itself strong activity evidence. It must
    # resolve a single-origin well instead of leaving it in "growth pending"
    # solely because an intermediate T1/T2 candidate had lower confidence.
    high_confidence = bool(
        single_origin
        and growth_active
        and (confidence >= 0.82 or late_growth)
    )
    # Multiplicity in T1/T2 is direct activity evidence even when the
    # individual frame confidence is below the high-confidence reporting
    # threshold.  A single T0 cell followed by one touching doublet or a
    # 3+ cluster is therefore active, not "growth pending".
    if growth_active:
        status = "single_active"
    elif single_origin:
        status = "single_not_divided"
    elif multi_origin:
        status = "multi_origin"
    elif counts["T0"] == 0 and max(counts["T1"], counts["T2"]) > 0:
        status = "t0_missing_late_cells"
    elif max(counts.values()) == 0:
        status = "no_cell_growth"
    else:
        status = "t0_missing_late_cells"
    return status, single_origin, multi_origin, growth_active, high_confidence


def _nonmaximum_objects(local: pd.DataFrame) -> pd.DataFrame:
    if local.empty:
        return local
    ordered = local.assign(
        _manual=local.get(
            "manual_override", pd.Series(False, index=local.index)
        ).astype(bool).astype(int),
        _cell=local["screen_label"].isin(CELL_LABELS).astype(int),
    ).sort_values(
        ["_manual", "_cell", "screen_confidence"], ascending=False
    )
    kept: list[int] = []
    centers: list[tuple[float, float]] = []
    for index, row in ordered.iterrows():
        x, y = float(row["x_px"]), float(row["y_px"])
        radius = max(10.0, min(28.0, float(row.get("diameter_px", 12.0)) * 0.55))
        if any(np.hypot(x - px, y - py) <= radius for px, py in centers):
            continue
        kept.append(index)
        centers.append((x, y))
    return ordered.loc[kept].drop(columns=["_manual", "_cell"])


def _representative_center(objects: pd.DataFrame) -> tuple[float, float] | None:
    cells = objects[objects["screen_label"].isin(CELL_LABELS)]
    if cells.empty:
        return None
    coordinates = cells[["x_px", "y_px"]].to_numpy(float)
    weights = np.asarray([CELL_UNITS[label] for label in cells["screen_label"]], dtype=float)
    if len(cells) == 1:
        return float(coordinates[0, 0]), float(coordinates[0, 1])
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    density = ((distances <= 420.0) * weights[None, :]).sum(axis=1)
    anchor = int(np.argmax(density))
    nearby = distances[anchor] <= 420.0
    center = np.average(coordinates[nearby], axis=0, weights=weights[nearby])
    return float(center[0]), float(center[1])


def _well_sort_key(well: str) -> tuple[str, int]:
    normalized = str(well).upper()
    try:
        return normalized[0], int(normalized[1:])
    except (IndexError, ValueError):
        return normalized, 0


def _merge_selected_well_rows(
    previous: pd.DataFrame,
    updated: pd.DataFrame,
    selected_wells: set[str],
    *,
    well_column: str = "well",
) -> pd.DataFrame:
    """Replace selected wells while preserving results for the rest of the plate."""
    normalized = {str(well).upper() for well in selected_wells}
    if previous.empty:
        merged = updated.copy()
    else:
        keep = ~previous[well_column].astype(str).str.upper().isin(normalized)
        merged = pd.concat([previous.loc[keep], updated], ignore_index=True, sort=False)
    if merged.empty:
        return merged
    order = merged[well_column].astype(str).map(_well_sort_key)
    return merged.iloc[sorted(range(len(merged)), key=lambda index: order.iloc[index])].reset_index(
        drop=True
    )


def build_well_screening(
    config: dict[str, Any],
    database: str | Path,
    selected_wells: set[str] | None = None,
) -> dict[str, Any]:
    source = _latest_prediction_source(config)
    if not source.exists():
        raise ValueError("Integrated predictions are unavailable.")
    output_path = artifact_path(config, "predictions", "latest_well_screening.csv")
    # An incremental update needs a complete plate result to merge into. Fall back
    # to a full build when the output has not been generated yet.
    selected_well_filter = (
        {str(well).upper() for well in selected_wells}
        if selected_wells and output_path.exists()
        else None
    )
    predictions = pd.read_csv(source, low_memory=False)
    round_id = (
        str(predictions.iloc[0]["integrated_round_id"])
        if len(predictions) and "integrated_round_id" in predictions
        else ""
    )
    if selected_well_filter is not None:
        predictions = predictions[
            predictions["well"].astype(str).str.upper().isin(selected_well_filter)
        ].copy()
    predictions["screen_label"] = predictions["integrated_label"].astype(str)
    predictions["screen_confidence"] = predictions["integrated_confidence"].astype(float)
    predictions["human_reviewed"] = False
    reviews = _latest_reviews(database)
    if not reviews.empty:
        lookup = reviews.set_index("candidate_id")["reviewed_label"]
        mapped = predictions["candidate_id"].astype(str).map(lookup)
        mask = mapped.notna()
        predictions.loc[mask, "screen_label"] = mapped[mask].astype(str)
        predictions.loc[mask, "screen_confidence"] = 1.0
        predictions.loc[mask, "human_reviewed"] = True

    manual = _manual_missed_objects(database, round_id)
    if not manual.empty:
        if selected_well_filter is not None:
            manual = manual[
                manual["well"].astype(str).str.upper().isin(selected_well_filter)
            ].copy()
        manual = manual[
            ~manual["candidate_id"].astype(str).isin(
                predictions["candidate_id"].astype(str)
            )
        ]
        predictions = pd.concat(
            [predictions, manual], ignore_index=True, sort=False
        )

    # V2 instance ownership is authoritative.  Applying the old circle/NMS
    # pass on top of V2 can re-introduce a wall residual or split an instance;
    # use the same visibility contract as the review UI instead.  Older V1
    # tables still use the established hierarchy suppression path.
    if "v2_instance_id" in predictions.columns:
        predictions = _visible_v2_rows(predictions)
        predictions["is_hierarchy_suppressed"] = False
        predictions["is_duplicate_suppressed"] = False
    else:
        predictions = suppress_nested_single_candidates(
            predictions,
            config,
            label_column="screen_label",
            confidence_column="screen_confidence",
        )
    predictions = predictions[
        ~predictions["is_hierarchy_suppressed"]
        & ~predictions["is_duplicate_suppressed"]
    ].copy()

    # Formal screening never exposes invalid material or wall candidates.
    predictions = predictions[
        predictions["timepoint"].isin(["T0", "T1", "T2"])
        & ~predictions["screen_label"].isin(
            ["invalid", "unmarked", "suppressed"]
        )
    ].copy()
    images = pd.read_csv(artifact_path(config, "manifests", "images.csv"))
    excluded = {
        str(value).upper()
        for value in config.get("review_queue", {}).get("excluded_wells", [])
    }
    image_wells = sorted(
        set(images["well"].astype(str).str.upper()) - excluded,
        key=_well_sort_key,
    )
    if selected_well_filter is not None:
        image_wells = [well for well in image_wells if well in selected_well_filter]
    ensure_well_screening_review_table(database)
    ensure_well_timepoint_cell_count_review_table(database)
    late_review_lookup = _late_growth_reviews(database)
    automatic_late_lookup = _automatic_late_growth(config)
    with sqlite3.connect(database) as connection:
        decisions = pd.read_sql_query("SELECT * FROM well_screening_reviews", connection)
        cell_count_reviews = pd.read_sql_query(
            "SELECT well, timepoint, cell_count FROM well_timepoint_cell_count_reviews",
            connection,
        )
    decision_lookup = (
        decisions.set_index("well").to_dict(orient="index") if not decisions.empty else {}
    )
    cell_count_review_lookup = {
        (str(row.well).upper(), str(row.timepoint).upper()): max(0, int(row.cell_count))
        for row in cell_count_reviews.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    object_rows: list[pd.DataFrame] = []
    for well in image_wells:
        local = predictions[predictions["well"].astype(str).str.upper() == well]
        available_timepoints = set(
            images[
                (images["well"].astype(str).str.upper() == well)
                & (images["decode_status"] == "ok")
            ]["timepoint"].astype(str)
        )
        by_timepoint: dict[str, pd.DataFrame] = {}
        centers: dict[str, tuple[float, float] | None] = {}
        counts: dict[str, int] = {}
        count_sources: dict[str, str] = {}
        for timepoint in ("T0", "T1", "T2"):
            selected = _nonmaximum_objects(local[local["timepoint"] == timepoint])
            by_timepoint[timepoint] = selected
            # V2 dead-cell inference has been retired.  Keep the legacy
            # columns in the screening output for schema compatibility, but
            # never use them to remove a counted cell.
            suspected_dead = pd.Series(False, index=selected.index)
            active_selected = selected.loc[~suspected_dead]
            cells = active_selected[active_selected["screen_label"].isin(CELL_LABELS)]
            automatic_count = int(sum(CELL_UNITS[label] for label in cells["screen_label"]))
            override = cell_count_review_lookup.get((well, timepoint))
            counts[timepoint] = automatic_count if override is None else override
            count_sources[timepoint] = "automatic" if override is None else "human"
            centers[timepoint] = _representative_center(active_selected)
            if not selected.empty:
                object_rows.append(selected.assign(screen_well=well))

        t0_suspected_dead = pd.Series(False, index=by_timepoint["T0"].index)
        t0_cells = by_timepoint["T0"][
            by_timepoint["T0"]["screen_label"].isin(CELL_LABELS)
            & ~t0_suspected_dead
        ]
        t0_uncertain = by_timepoint["T0"][by_timepoint["T0"]["screen_label"] == "uncertain"]
        evidence = pd.concat([by_timepoint[tp] for tp in ("T0", "T1", "T2")])
        evidence_suspected_dead = pd.Series(False, index=evidence.index)
        cell_evidence = evidence[
            evidence["screen_label"].isin(CELL_LABELS)
            & ~evidence_suspected_dead
        ]
        confidence = float(cell_evidence["screen_confidence"].min()) if not cell_evidence.empty else 0.0
        reviewed_fraction = float(evidence["human_reviewed"].mean()) if not evidence.empty else 0.0
        late_decisions = {
            timepoint: str(
                late_review_lookup.get((well, timepoint), {}).get(
                    "decision",
                    automatic_late_lookup.get((well, timepoint), {}).get(
                        "automatic_decision", "pending"
                    ),
                )
            )
            for timepoint in LATE_GROWTH_TIMEPOINTS
        }
        late_growth_status, skip_deep_search = late_growth_gate(
            available_timepoints,
            late_decisions,
        )
        (
            base_status,
            single_origin,
            multi_origin,
            growth_active,
            high_confidence,
        ) = _classify_well_status(
            counts,
            t0_cell_instances=len(t0_cells),
            t0_has_uncertain=not t0_uncertain.empty,
            confidence=confidence,
            late_growth_status=late_growth_status,
        )
        status = "no_cell_growth" if skip_deep_search else base_status
        if skip_deep_search:
            high_confidence = False
        last_center = None
        roi: dict[str, dict[str, float] | None] = {}
        for timepoint in ("T0", "T1", "T2"):
            center = centers[timepoint] or last_center
            if center:
                last_center = center
                roi[timepoint] = {"x": center[0], "y": center[1], "size": 900.0}
            else:
                roi[timepoint] = None
        for timepoint in LATE_GROWTH_TIMEPOINTS:
            automatic_late = automatic_late_lookup.get((well, timepoint), {})
            dense_x = automatic_late.get("dense_center_x")
            dense_y = automatic_late.get("dense_center_y")
            use_dense_center = (
                automatic_late.get("automatic_decision") == "obvious_growth"
                and dense_x is not None
                and dense_y is not None
                and not pd.isna(dense_x)
                and not pd.isna(dense_y)
            )
            if use_dense_center:
                roi[timepoint] = {
                    "x": float(dense_x), "y": float(dense_y), "size": 1100.0
                }
            else:
                roi[timepoint] = (
                    {"x": last_center[0], "y": last_center[1], "size": 1100.0}
                    if last_center else None
                )
        late_sources = {
            timepoint: (
                "human"
                if (well, timepoint) in late_review_lookup
                else "automatic"
                if (well, timepoint) in automatic_late_lookup
                else "pending"
            )
            for timepoint in LATE_GROWTH_TIMEPOINTS
        }
        has_debris = bool((evidence["screen_label"] == "debris").any())
        decision = decision_lookup.get(well, {})
        rows.append({
            "well": well,
            "screening_status": status,
            "base_screening_status": base_status,
            "high_confidence_single_active": high_confidence,
            "single_cell_origin": bool(single_origin),
            "multi_cell_origin": bool(multi_origin),
            "growth_division_active": bool(growth_active),
            "t0_cell_units": counts["T0"],
            "t1_cell_units": counts["T1"],
            "t2_cell_units": counts["T2"],
            "t0_cell_units_source": count_sources["T0"],
            "t1_cell_units_source": count_sources["T1"],
            "t2_cell_units_source": count_sources["T2"],
            "t0_cell_instances": int(len(t0_cells)),
            "t0_suspected_dead_instances": int(t0_suspected_dead.sum()),
            "suspected_dead_cell": bool(evidence_suspected_dead.any()),
            "t0_has_uncertain": bool(not t0_uncertain.empty),
            "early_timepoints_complete": all(
                timepoint in available_timepoints for timepoint in ("T0", "T1", "T2")
            ),
            "early_division_evidence": bool(
                counts["T0"] == 1 and max(counts["T1"], counts["T2"]) >= 2
            ),
            # Reserved for the learned early-object association stage. A
            # conflict must force the final gated report into manual review.
            "temporal_link_conflict": False,
            "image_quality_failure": False,
            "has_debris": has_debris,
            "t3_available": "T3" in available_timepoints,
            "t4_available": "T4" in available_timepoints,
            "t3_growth_decision": late_decisions["T3"],
            "t4_growth_decision": late_decisions["T4"],
            "t3_growth_source": late_sources["T3"],
            "t4_growth_source": late_sources["T4"],
            "t3_growth_search_stage": automatic_late_lookup.get((well, "T3"), {}).get("search_stage", ""),
            "t4_growth_search_stage": automatic_late_lookup.get((well, "T4"), {}).get("search_stage", ""),
            "late_growth_status": late_growth_status,
            "skip_deep_search": bool(skip_deep_search),
            "deep_search_required": not bool(skip_deep_search),
            "screening_confidence": confidence,
            "reviewed_fraction": reviewed_fraction,
            "review_decision": decision.get("decision", "unclassified"),
            "review_notes": decision.get("notes", ""),
            "roi_json": json.dumps(roi, ensure_ascii=False),
        })
    result = pd.DataFrame(rows)
    object_result = (
        pd.concat(object_rows, ignore_index=True) if object_rows else pd.DataFrame()
    )
    objects_path = artifact_path(config, "predictions", "latest_screening_objects.csv")
    if selected_well_filter is not None:
        previous = pd.read_csv(output_path, low_memory=False)
        result = _merge_selected_well_rows(
            previous, result, selected_well_filter
        )
        previous_objects = (
            pd.read_csv(objects_path, low_memory=False)
            if objects_path.exists()
            else pd.DataFrame()
        )
        object_result = _merge_selected_well_rows(
            previous_objects,
            object_result,
            selected_well_filter,
            well_column="screen_well",
        )
    result.to_csv(output_path, index=False, encoding="utf-8")
    result[result["deep_search_required"]].to_csv(
        artifact_path(config, "predictions", "latest_deep_search_queue.csv"),
        index=False,
        encoding="utf-8",
    )
    if not object_result.empty:
        object_result.to_csv(objects_path, index=False, encoding="utf-8")
    elif objects_path.exists():
        objects_path.unlink()
    summary = {
        "well_count": int(len(result)),
        "high_confidence_single_active": int(result["high_confidence_single_active"].sum()),
        "single_not_divided": int((result["screening_status"] == "single_not_divided").sum()),
        "multi_origin": int((result["screening_status"] == "multi_origin").sum()),
        "pending_review": int((result["review_decision"] == "pending").sum()),
        "unclassified_review": int((result["review_decision"] == "unclassified").sum()),
        "late_growth_no_growth": int((result["late_growth_status"] == "no_growth").sum()),
        "late_growth_pending": int((result["late_growth_status"] == "pending").sum()),
        "deep_search_skipped": int(result["skip_deep_search"].sum()),
        "deep_search_required": int(result["deep_search_required"].sum()),
        "output": str(artifact_path(config, "predictions", "latest_well_screening.csv")),
        "deep_search_queue": str(
            artifact_path(config, "predictions", "latest_deep_search_queue.csv")
        ),
    }
    artifact_path(config, "predictions", "latest_well_screening_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
