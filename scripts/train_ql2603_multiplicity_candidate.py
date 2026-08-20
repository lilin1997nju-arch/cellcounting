"""Train an isolated QL2603 multiplicity-head candidate with a baseline backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from cellvision.config import artifact_path, load_config
from cellvision.multiplicity import train_multiplicity_classifier


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ql2603_joint_training.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--baseline-checkpoint",
        default="artifacts/models/multiplicity_classifier.pt",
    )
    parser.add_argument("--hard-example-weight", type=float, default=4.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir = output_dir / "candidate"
    backup_dir = output_dir / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    baseline = Path(args.baseline_checkpoint).expanduser().resolve()
    if not baseline.exists():
        raise FileNotFoundError(baseline)
    shutil.copy2(baseline, backup_dir / baseline.name)
    baseline_report = baseline.with_suffix(".json")
    if baseline_report.exists():
        shutil.copy2(baseline_report, backup_dir / baseline_report.name)

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    config.setdefault("multiplicity", {})["output_directory"] = str(model_dir)
    config["multiplicity"]["corrected_single_doublet_weight"] = float(
        args.hard_example_weight
    )
    database = artifact_path(config, "annotations", "annotations.db")
    report = train_multiplicity_classifier(config, database)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "validation_holdouts": config.get("validation_holdout_sources", []),
        "hard_example_weight": float(args.hard_example_weight),
        "baseline": {
            "path": str(baseline),
            "backup": str(backup_dir / baseline.name),
            "sha256": _sha256(baseline),
        },
        "candidate": {
            "path": report["checkpoint"],
            "sha256": _sha256(Path(report["checkpoint"])),
        },
        "training": report,
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
