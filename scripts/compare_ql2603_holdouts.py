from __future__ import annotations

import json
import shutil
from pathlib import Path

from cellvision.config import load_config
from cellvision.evaluate_v2 import write_v2_evaluation


ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "artifacts" / "backups" / "ql2603_training_round_20260811_pre"


def compare_plate(slug: str) -> dict[str, object]:
    config_path = ROOT / "configs" / "generated" / f"{slug}.yaml"
    config = load_config(config_path)
    plate_root = ROOT / "artifacts" / "projects" / "ql2603" / "plates" / slug
    predictions = plate_root / "predictions" / "latest_v2_predictions.csv"
    instance_predictions = plate_root / "predictions" / "latest_v2_pre_temporal_predictions.csv"
    baseline = BACKUP / slug / "predictions" / "latest_v2_predictions.csv"
    incremental_copy = predictions.with_name("latest_v2_predictions.incremental.csv")
    temporal_copy = predictions.with_name("latest_v2_predictions.temporal.csv")
    evaluation = plate_root / "evaluation"
    baseline_json = evaluation / "v2_metrics_baseline.json"
    incremental_json = evaluation / "v2_metrics_incremental.json"

    if not predictions.exists():
        raise FileNotFoundError(predictions)
    if not instance_predictions.exists():
        raise FileNotFoundError(instance_predictions)
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    evaluation.mkdir(parents=True, exist_ok=True)

    shutil.copy2(predictions, temporal_copy)
    shutil.copy2(instance_predictions, incremental_copy)
    try:
        shutil.copy2(baseline, predictions)
        write_v2_evaluation(config)
        shutil.copy2(evaluation / "v2_metrics.json", baseline_json)
        shutil.copy2(incremental_copy, predictions)
        write_v2_evaluation(config)
        shutil.copy2(evaluation / "v2_metrics.json", incremental_json)
    finally:
        shutil.copy2(temporal_copy, predictions)
        incremental_copy.unlink(missing_ok=True)
        temporal_copy.unlink(missing_ok=True)

    baseline_result = json.loads(baseline_json.read_text(encoding="utf-8"))
    incremental_result = json.loads(incremental_json.read_text(encoding="utf-8"))
    return {
        "plate_id": slug,
        "baseline": baseline_result,
        "incremental": incremental_result,
    }


def main() -> None:
    output = {
        slug: compare_plate(slug)
        for slug in ("ql2603-t1-2", "ql2603-t4-2")
    }
    report = ROOT / "artifacts" / "v2" / "runs" / "ql2603_holdout_v2_comparison.json"
    report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for slug, result in output.items():
        print(slug)
        for name in ("baseline", "incremental"):
            metrics = result[name]
            print(
                name,
                "recall=", metrics.get("target_level_recall"),
                "wall_fp=", metrics.get("wall_false_positive_rate"),
                "near_wall=", metrics.get("near_wall_cell_recall"),
                "multiplicity=", metrics.get("single_doublet_cluster_confusion"),
            )
    print(report)


if __name__ == "__main__":
    main()
