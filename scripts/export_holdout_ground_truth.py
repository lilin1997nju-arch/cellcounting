from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.database) as connection:
        reviews = pd.read_sql_query(
            """
            SELECT candidate_id, predicted_label AS reviewed_prediction,
                   reviewed_label, decision, reviewer, updated_at
            FROM integrated_training_reviews
            WHERE round_id = ?
            ORDER BY updated_at
            """,
            connection,
            params=(args.round_id,),
        ).drop_duplicates("candidate_id", keep="last")
        missed = pd.read_sql_query(
            """
            SELECT candidate_id, well, timepoint, x_px, y_px, diameter_px,
                   reviewed_label, reviewer, updated_at
            FROM quick_missed_objects
            WHERE round_id = ?
            ORDER BY updated_at
            """,
            connection,
            params=(args.round_id,),
        ).drop_duplicates("candidate_id", keep="last")
    predictions = pd.read_csv(args.predictions, low_memory=False)
    columns = [
        "candidate_id", "well", "timepoint", "x_px", "y_px",
        "diameter_px", "integrated_label", "integrated_confidence",
        "cell_probability", "debris_probability", "invalid_probability",
        "single_probability", "touching_doublet_probability",
        "cluster_3plus_probability",
    ]
    audited = reviews.merge(
        predictions[[column for column in columns if column in predictions]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"integrated_label": "baseline_label"})
    audited.to_csv(output / "audited_candidates.csv", index=False, encoding="utf-8")
    missed.to_csv(output / "manual_missed_objects.csv", index=False, encoding="utf-8")
    summary = {
        "round_id": args.round_id,
        "audited_candidate_count": int(len(audited)),
        "manual_missed_count": int(len(missed)),
        "reviewed_label_counts": {
            str(key): int(value)
            for key, value in audited["reviewed_label"].value_counts().items()
        },
        "manual_missed_label_counts": {
            str(key): int(value)
            for key, value in missed["reviewed_label"].value_counts().items()
        },
        "source_predictions": str(Path(args.predictions).resolve()),
        "source_database": str(Path(args.database).resolve()),
    }
    (output / "ground_truth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
