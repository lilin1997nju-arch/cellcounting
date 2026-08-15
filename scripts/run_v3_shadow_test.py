from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from cellvision.config import load_config
from cellvision.v2_temporal_inference import infer_v2_temporal_evidence


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def run_shadow_test(
    config_paths: list[str],
    output_root: str,
    state_fusion: str = "shadow",
) -> Path:
    if state_fusion not in {"shadow", "active"}:
        raise ValueError(f"Unsupported V3 state fusion mode: {state_fusion}")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "mode": f"v3_{state_fusion}",
        "state_fusion": state_fusion,
        "plates": [],
    }
    for config_path in config_paths:
        started = time.perf_counter()
        config = load_config(config_path)
        plate_id = str(config.get("experiment", {}).get("plate_id", Path(config_path).stem))
        source_root = Path(config["paths"]["artifact_root"])
        source = source_root / "predictions" / "latest_v2_predictions.csv"
        if not source.exists():
            raise FileNotFoundError(f"Prediction file does not exist for {plate_id}: {source}")
        plate_root = root / _safe_name(plate_id)
        (plate_root / "predictions").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, plate_root / "predictions" / "latest_v2_predictions.csv")

        config["paths"]["artifact_root"] = str(plate_root)
        behavior = config.setdefault("v3_temporal_behavior", {})
        behavior["enabled"] = True
        behavior["state_fusion"] = state_fusion
        behavior["backend"] = "heuristic_behavior_v1"
        output = infer_v2_temporal_evidence(config)
        summary_path = output.with_name("latest_v2_temporal_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(output, low_memory=False)
        v3 = predictions[predictions["v3_track_behavior"].ne("disabled")]
        temporal_candidates = predictions["v2_is_temporal_candidate"].fillna(False).astype(bool)
        legacy_changed = (
            predictions["integrated_label"].astype(str)
            != predictions["v2_pre_temporal_integrated_label"].astype(str)
        )
        plate_report = {
            "plate_id": plate_id,
            "config": str(Path(config_path).resolve()),
            "input_predictions": str(source.resolve()),
            "output_predictions": str(output.resolve()),
            "summary": summary,
            "rows": int(len(predictions)),
            "temporal_candidate_rows": int(temporal_candidates.sum()),
            "v3_track_count": int(v3["v3_track_id"].replace("", pd.NA).nunique()),
            "v3_candidate_coverage": float(len(v3) / max(int(temporal_candidates.sum()), 1)),
            "v3_rows": int(len(v3)),
            "legacy_final_changed_rows": int(legacy_changed.sum()),
            "legacy_label_counts": predictions["v2_pre_temporal_integrated_label"].value_counts().to_dict(),
            "legacy_final_label_counts": predictions["integrated_label"].value_counts().to_dict(),
            "v3_track_behavior_counts": v3["v3_track_behavior"].value_counts().to_dict(),
            "v3_unified_label_counts": v3["v3_track_conclusion"].replace("", pd.NA).value_counts().to_dict()
            if "v3_track_conclusion" in v3
            else {},
            "v3_wall_origin_counts": v3["v3_wall_origin"].value_counts().to_dict(),
            "v3_reason_counts": v3["v3_reason"].value_counts().to_dict(),
            "v3_would_change_rows": int(v3["v3_would_change"].fillna(False).astype(bool).sum()),
            "v3_cell_to_debris_rows": int(v3["v3_cell_to_debris_candidate"].fillna(False).astype(bool).sum()),
            "v3_semantic_degradation_rows": int(v3["v3_semantic_degradation"].fillna(False).astype(bool).sum()),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        report["plates"].append(plate_report)
        (root / f"{_safe_name(plate_id)}.summary.json").write_text(
            json.dumps(plate_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"plate_id": plate_id, **{key: plate_report[key] for key in ("rows", "v3_rows", "v3_would_change_rows", "v3_cell_to_debris_rows", "v3_semantic_degradation_rows", "elapsed_seconds")}}, ensure_ascii=False), flush=True)

    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(report_path), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 temporal behavior on existing plate predictions")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/default.yaml", "configs/ql2202_validation.yaml"],
    )
    parser.add_argument(
        "--output",
        default="artifacts/v3_shadow_tests/20260809",
    )
    parser.add_argument(
        "--state-fusion",
        choices=["shadow", "active"],
        default="shadow",
        help="Keep V3 as an audit-only proposal or apply its frame-level conclusion",
    )
    args = parser.parse_args()
    run_shadow_test(args.configs, args.output, args.state_fusion)


if __name__ == "__main__":
    main()
