from __future__ import annotations

import json
import shutil
from pathlib import Path

from cellvision.config import load_config
from cellvision.evaluate_v2 import write_v2_evaluation


ROOT = Path(__file__).resolve().parents[1]
PLATE = ROOT / "artifacts" / "training" / "a12_22"
BACKUP = ROOT / "artifacts" / "backups" / "ql2603_training_round_20260811_pre" / "a12_22"
INCREMENTAL = ROOT / "artifacts" / "v2" / "runs" / "ql2603_v2_scratch_comparison" / "A12-22_plate_1" / "incremental_temporal_predictions.csv"
SCRATCH = ROOT / "artifacts" / "v2" / "runs" / "ql2603_v2_scratch_comparison" / "A12-22_plate_1" / "scratch_temporal_predictions.csv"
OUTPUT = ROOT / "artifacts" / "v2" / "runs" / "ql2603_a12_22_v2_comparison.json"


def main() -> None:
    config = load_config(ROOT / "configs" / "a12_22_training.yaml")
    predictions = PLATE / "predictions" / "latest_v2_predictions.csv"
    evaluation = PLATE / "evaluation" / "v2_metrics.json"
    baseline_predictions = BACKUP / "latest_v2_predictions.csv"
    baseline_metrics = BACKUP / "v2_metrics.json"
    saved = PLATE / "predictions" / "latest_v2_predictions.active.csv"
    results = {}
    shutil.copy2(predictions, saved)
    try:
        for name, source in (
            ("current_baseline", baseline_predictions),
            ("incremental", INCREMENTAL),
            ("scratch", SCRATCH),
        ):
            shutil.copy2(source, predictions)
            write_v2_evaluation(config)
            results[name] = json.loads(evaluation.read_text(encoding="utf-8"))
    finally:
        shutil.copy2(saved, predictions)
        saved.unlink(missing_ok=True)
        shutil.copy2(baseline_metrics, evaluation)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in results.items():
        confusion = result["single_doublet_cluster_confusion"]
        print(
            name,
            "recall=", result["target_level_recall"],
            "doublet_recall=", confusion.get("touching_doublet_recall"),
            "doublet_f1=", confusion.get("touching_doublet_f1"),
            "near_wall=", result["near_wall_cell_recall"],
        )
    print(OUTPUT)


if __name__ == "__main__":
    main()
