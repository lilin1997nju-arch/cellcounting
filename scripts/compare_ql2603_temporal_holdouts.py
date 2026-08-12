from __future__ import annotations

import json
import shutil
from pathlib import Path

from cellvision.config import load_config
from cellvision.evaluate_v2 import write_v2_evaluation


ROOT = Path(__file__).resolve().parents[1]
OLD_ROUNDS = {
    "ql2603-t1-2": "v2-temporal-round-20260806-085023",
    "ql2603-t4-2": "v2-temporal-round-20260806-102601",
}


def compare_plate(slug: str, old_round: str) -> dict[str, object]:
    config = load_config(ROOT / "configs" / "generated" / f"{slug}.yaml")
    plate_root = ROOT / "artifacts" / "projects" / "ql2603" / "plates" / slug
    predictions = plate_root / "predictions" / "latest_v2_predictions.csv"
    baseline = plate_root / "predictions" / old_round / "predictions.csv"
    incremental_copy = predictions.with_name("latest_v2_predictions.incremental.csv")
    evaluation = plate_root / "evaluation"
    baseline_json = evaluation / "v2_metrics_temporal_baseline.json"
    incremental_json = evaluation / "v2_metrics_temporal_incremental.json"
    evaluation.mkdir(parents=True, exist_ok=True)
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    shutil.copy2(predictions, incremental_copy)
    try:
        shutil.copy2(baseline, predictions)
        write_v2_evaluation(config)
        shutil.copy2(evaluation / "v2_metrics.json", baseline_json)
        shutil.copy2(incremental_copy, predictions)
        write_v2_evaluation(config)
        shutil.copy2(evaluation / "v2_metrics.json", incremental_json)
    finally:
        shutil.copy2(incremental_copy, predictions)
        incremental_copy.unlink(missing_ok=True)
    return {
        "baseline": json.loads(baseline_json.read_text(encoding="utf-8")),
        "incremental": json.loads(incremental_json.read_text(encoding="utf-8")),
    }


def main() -> None:
    report = {
        slug: compare_plate(slug, old_round)
        for slug, old_round in OLD_ROUNDS.items()
    }
    path = ROOT / "artifacts" / "v2" / "runs" / "ql2603_temporal_holdout_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for slug, result in report.items():
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
    print(path)


if __name__ == "__main__":
    main()
