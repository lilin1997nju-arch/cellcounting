"""Activate a V3 temporal run in the real project review application.

The shadow runner deliberately writes to an isolated run directory.  This
script publishes those per-plate predictions under the project artifact roots
as ``latest_v3_predictions.csv`` and carries existing human labels into the
new round without replacing the previous V2 files.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cellvision.multiplicity import (
    INTEGRATED_REVIEW_LABELS,
    carry_forward_integrated_reviews,
)
from cellvision.config import artifact_path, load_config
from cellvision.gated_screening import build_gated_plate_report
from cellvision.well_screening import build_well_screening


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def latest_human_labels(database: Path, round_id: str) -> dict[str, str]:
    """Return the latest valid human label for each previous candidate."""
    if not database.exists():
        return {}
    with sqlite3.connect(database) as connection:
        previous = pd.read_sql_query(
            """
            SELECT integrated_review_id, candidate_id, reviewed_label, updated_at
            FROM integrated_training_reviews
            WHERE round_id <> ?
            ORDER BY updated_at, integrated_review_id
            """,
            connection,
            params=(round_id,),
        )
        try:
            missed = pd.read_sql_query(
                """
                SELECT quick_missed_id AS integrated_review_id,
                       candidate_id, reviewed_label, updated_at
                FROM quick_missed_objects
                WHERE round_id <> ?
                """,
                connection,
                params=(round_id,),
            )
            previous = pd.concat(
                [previous, missed], ignore_index=True, sort=False
            )
        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            pass
    if previous.empty:
        return {}
    previous = previous.sort_values(
        ["updated_at", "integrated_review_id"]
    ).drop_duplicates("candidate_id", keep="last")
    previous = previous[
        previous["reviewed_label"].astype(str).isin(INTEGRATED_REVIEW_LABELS)
    ]
    return dict(
        zip(
            previous["candidate_id"].astype(str),
            previous["reviewed_label"].astype(str),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="artifacts/v3_shadow_tests/20260809_2603_all20_active/report.json",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/projects/ql2603/project.json",
    )
    args = parser.parse_args()

    report_path = resolve(args.report)
    manifest_path = resolve(args.manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plates_by_id = {
        str(item.get("plate_id")): item for item in report.get("plates", [])
    }
    activated: list[dict[str, object]] = []
    for plate in manifest.get("plates", []):
        plate_id = f"QL2603_{plate.get('board_id', '')}"
        result = plates_by_id.get(plate_id)
        if result is None:
            raise RuntimeError(f"V3 report has no result for {plate_id}")
        source = resolve(str(result["output_predictions"]))
        target_root = resolve(str(plate["artifact_root"]))
        target_predictions = target_root / "predictions"
        target_predictions.mkdir(parents=True, exist_ok=True)
        published = target_predictions / "latest_v3_predictions.csv"

        source_summary = source.with_name("latest_v2_temporal_summary.json")
        if source_summary.exists():
            shutil.copy2(
                source_summary,
                target_predictions / "latest_v3_temporal_summary.json",
            )

        predictions = pd.read_csv(source, low_memory=False)
        round_id = str(predictions.iloc[0]["integrated_round_id"])
        database = target_root / "annotations" / "annotations.db"
        human_labels = latest_human_labels(database, round_id)
        candidate_ids = predictions["candidate_id"].astype(str)
        override_labels = candidate_ids.map(human_labels)
        override_mask = (
            predictions["integrated_label"].astype(str).eq("unmarked")
            & override_labels.notna()
        )
        if override_mask.any():
            predictions.loc[override_mask, "integrated_label"] = override_labels[
                override_mask
            ]
        predictions.to_csv(published, index=False)
        carried_reviews = carry_forward_integrated_reviews(
            database,
            round_id,
            predictions,
        )
        plate_config = load_config(resolve(str(plate["config"])))
        screening_summary = build_well_screening(plate_config, database)
        gated_summary = None
        gated_settings = plate_config.get("gated_report", {})
        endpoint_csv = gated_settings.get("endpoint_csv") or gated_settings.get(
            "day14_csv"
        )
        if (
            endpoint_csv
            and gated_settings.get("group_id")
            and gated_settings.get("output_dir")
        ):
            endpoint_timepoint = str(
                gated_settings.get("endpoint_timepoint", "T4")
            ).upper()
            endpoint_day_label = str(
                gated_settings.get("endpoint_day_label", "Day14")
            )
            with sqlite3.connect(database) as connection:
                try:
                    late_rows = connection.execute(
                        "SELECT well, decision FROM late_growth_reviews WHERE timepoint = ?",
                        (endpoint_timepoint,),
                    ).fetchall()
                except sqlite3.OperationalError:
                    late_rows = []
            gated_summary = build_gated_plate_report(
                endpoint_csv,
                gated_settings["group_id"],
                gated_settings["output_dir"],
                early_screening_csv=artifact_path(
                    plate_config,
                    "predictions",
                    "latest_well_screening.csv",
                ),
                sessions_csv=gated_settings.get("sessions_csv"),
                locate_day7=False,
                day14_growth_overrides={
                    str(well).upper(): str(decision)
                    for well, decision in late_rows
                },
                endpoint_day_label=endpoint_day_label,
            )
        activation = {
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": report_path.parent.name,
            "mode": report.get("mode", "v3_active"),
            "state_fusion": report.get("state_fusion", "active"),
            "checkpoint": report.get("checkpoint", ""),
            "plate_id": plate_id,
            "round_id": round_id,
            "source_predictions": str(source),
            "published_predictions": str(published.resolve()),
            "rows": int(len(predictions)),
            "carried_forward_reviews": int(carried_reviews),
            "human_label_overrides": int(override_mask.sum()),
            "screening_well_count": int(screening_summary.get("well_count", 0)),
            "gated_report_refreshed": gated_summary is not None,
        }
        (target_predictions / "v3_activation.json").write_text(
            json.dumps(activation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        activated.append(activation)
        print(json.dumps(activation, ensure_ascii=False), flush=True)

    activation_report = {
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": report_path.parent.name,
        "mode": report.get("mode", "v3_active"),
        "state_fusion": report.get("state_fusion", "active"),
        "plate_count": len(activated),
        "plates": activated,
    }
    output = report_path.parent / "project_activation.json"
    output.write_text(
        json.dumps(activation_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(str(output.resolve()), flush=True)


if __name__ == "__main__":
    main()
