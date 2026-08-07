from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLATES = {
    "QL11111": ROOT / "artifacts/evaluation/v2_metrics.json",
    "QL2202": ROOT / "artifacts/validation/ql2202/evaluation/v2_metrics.json",
    "A12-22": ROOT / "artifacts/training/a12_22/evaluation/v2_metrics.json",
}


def main() -> None:
    plates = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in PLATES.items() if path.exists()}
    recalls = {name: value.get("target_level_recall") for name, value in plates.items()}
    finite = [value for value in recalls.values() if value is not None]
    report = {
        "algorithm_version": "v2",
        "primary_product_metric": "well-level active single-cell conclusion accuracy",
        "plate_metrics": plates,
        "cross_plate_generalization": {
            "target_recall_by_plate": recalls,
            "target_recall_range": max(finite) - min(finite) if finite else None,
            "interpretation": "A12-22 is external validation and was excluded from all V2 weight updates.",
        },
        "well_level_metric_status": "pending frozen human well-level activity labels; existing V1 growth decision intentionally preserved",
    }
    output = ROOT / "artifacts/v2/evaluation/all_plate_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
