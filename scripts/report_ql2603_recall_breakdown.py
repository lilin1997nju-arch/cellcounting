from __future__ import annotations

import json
import shutil
from pathlib import Path

from cellvision.config import artifact_path, load_config
from cellvision.evaluate_v2 import write_v2_evaluation


ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "artifacts" / "backups" / "ql2603_training_round_20260811_pre"
OUTPUT = ROOT / "artifacts" / "v2" / "runs" / "ql2603_recall_breakdown.json"


def evaluate_variants(config_path: Path, baseline: Path, incremental: Path, scratch: Path) -> dict[str, object]:
    config = load_config(config_path)
    predictions = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    evaluation = artifact_path(config, "evaluation", "v2_metrics.json")
    active_predictions = predictions.with_name("latest_v2_predictions.active.csv")
    active_metrics = evaluation.with_name("v2_metrics.active.json")
    shutil.copy2(predictions, active_predictions)
    shutil.copy2(evaluation, active_metrics)
    results: dict[str, object] = {}
    try:
        for name, source in (("current_baseline", baseline), ("incremental", incremental), ("scratch", scratch)):
            shutil.copy2(source, predictions)
            write_v2_evaluation(config)
            results[name] = json.loads(evaluation.read_text(encoding="utf-8"))
    finally:
        shutil.copy2(active_predictions, predictions)
        shutil.copy2(active_metrics, evaluation)
        active_predictions.unlink(missing_ok=True)
        active_metrics.unlink(missing_ok=True)
    return results


def main() -> None:
    variants = {
        "ql2603-t1-2": (
            ROOT / "configs/generated/ql2603-t1-2.yaml",
            BACKUP / "ql2603-t1-2/predictions/latest_v2_predictions.csv",
            ROOT / "artifacts/projects/ql2603/plates/ql2603-t1-2/predictions/v2-temporal-round-20260811-170809/predictions.csv",
            ROOT / "artifacts/v2/runs/ql2603_v2_scratch_comparison/QL2603_T1-2/scratch_temporal_predictions.csv",
        ),
        "ql2603-t4-2": (
            ROOT / "configs/generated/ql2603-t4-2.yaml",
            BACKUP / "ql2603-t4-2/predictions/latest_v2_predictions.csv",
            ROOT / "artifacts/projects/ql2603/plates/ql2603-t4-2/predictions/v2-temporal-round-20260811-170804/predictions.csv",
            ROOT / "artifacts/v2/runs/ql2603_v2_scratch_comparison/QL2603_T4-2/scratch_temporal_predictions.csv",
        ),
        "A12-22": (
            ROOT / "configs/a12_22_training.yaml",
            BACKUP / "a12_22/latest_v2_predictions.csv",
            ROOT / "artifacts/training/a12_22/predictions/v2-temporal-round-20260811-171611/predictions.csv",
            ROOT / "artifacts/v2/runs/ql2603_v2_scratch_comparison/A12-22_plate_1/scratch_temporal_predictions.csv",
        ),
    }
    report = {}
    for name, (config, baseline, incremental, scratch) in variants.items():
        report[name] = evaluate_variants(config, baseline, incremental, scratch)
        print(name)
        for variant, result in report[name].items():
            confusion = result["single_doublet_cluster_confusion"]
            print(
                variant,
                "overall=", result["target_level_recall"],
                "cell=", result["cell_level_recall"],
                "doublet_e2e=", confusion["touching_doublet_end_to_end_recall"],
                "doublet_conditional_f1=", confusion["touching_doublet_f1"],
                "doublet_e2e_f1=", confusion["touching_doublet_end_to_end_f1"],
            )
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
