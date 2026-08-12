from __future__ import annotations

import json
import shutil
from pathlib import Path

from cellvision.config import artifact_path, load_config
from cellvision.evaluate_v2 import write_v2_evaluation
from cellvision.v2_instance_inference import infer_v2_instances
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_CHECKPOINT = ROOT / "artifacts" / "v2" / "runs" / "v2-instance-20260811-171801" / "model.pt"
TEMPORAL_CHECKPOINT = ROOT / "artifacts" / "v2" / "runs" / "v2-temporal-20260811-171910" / "model.pt"
OUTPUT = ROOT / "artifacts" / "v2" / "runs" / "ql2603_v2_scratch_comparison"
CONFIGS = (
    ROOT / "configs" / "generated" / "ql2603-t1-2.yaml",
    ROOT / "configs" / "generated" / "ql2603-t4-2.yaml",
    ROOT / "configs" / "a12_22_training.yaml",
)


def evaluate_one(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    slug = str(config["experiment"]["plate_id"]).replace("/", "_")
    predictions = artifact_path(config, "predictions", "latest_v2_predictions.csv")
    pre_temporal = predictions.with_name("latest_v2_pre_temporal_predictions.csv")
    summary = predictions.with_suffix(".json")
    temporal_summary = predictions.with_name("latest_v2_temporal_summary.json")
    local = OUTPUT / slug
    local.mkdir(parents=True, exist_ok=True)
    saved = {
        "predictions": local / "incremental_temporal_predictions.csv",
        "pre_temporal": local / "incremental_instance_predictions.csv",
        "summary": local / "incremental_instance_summary.json",
        "temporal_summary": local / "incremental_temporal_summary.json",
    }
    for source, destination in (
        (predictions, saved["predictions"]),
        (pre_temporal, saved["pre_temporal"]),
        (summary, saved["summary"]),
        (temporal_summary, saved["temporal_summary"]),
    ):
        if source.exists():
            shutil.copy2(source, destination)
    try:
        infer_v2_instances(config, INSTANCE_CHECKPOINT)
        shutil.copy2(pre_temporal, local / "scratch_instance_predictions.csv")
        infer_v2_temporal_evidence(config, TEMPORAL_CHECKPOINT)
        shutil.copy2(predictions, local / "scratch_temporal_predictions.csv")
        write_v2_evaluation(config)
        metrics_path = local / "scratch_v2_metrics.json"
        shutil.copy2(artifact_path(config, "evaluation", "v2_metrics.json"), metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    finally:
        for source, destination in (
            (saved["predictions"], predictions),
            (saved["pre_temporal"], pre_temporal),
            (saved["summary"], summary),
            (saved["temporal_summary"], temporal_summary),
        ):
            if source.exists():
                shutil.copy2(source, destination)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for config_path in CONFIGS:
        result = evaluate_one(config_path)
        plate_id = str(result["plate_id"])
        report[plate_id] = result
        print(
            plate_id,
            "recall=", result.get("target_level_recall"),
            "wall_fp=", result.get("wall_false_positive_rate"),
            "near_wall=", result.get("near_wall_cell_recall"),
            "multiplicity=", result.get("single_doublet_cluster_confusion"),
        )
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
