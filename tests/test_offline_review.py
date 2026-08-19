import json
import sqlite3
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from cellvision.offline_review import (
    BUNDLE_FORMAT,
    build_offline_review_bundle,
    export_offline_review_results,
    import_offline_review_results,
)
from cellvision.review_storage import initialize_database


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    artifact = tmp_path / "artifact"
    (artifact / "predictions").mkdir(parents=True)
    (artifact / "manifests").mkdir()
    (artifact / "gated").mkdir()
    image_path = tmp_path / "C2.tif"
    Image.new("L", (100, 80), 128).save(image_path)
    pd.DataFrame([{
        "experiment_id": "exp", "plate_id": "plate", "well": "C2", "timepoint": "T0",
        "raw_image_path": str(image_path), "width_px": 100, "height_px": 80, "decode_status": "ok",
    }]).to_csv(artifact / "manifests" / "images.csv", index=False)
    pd.DataFrame([{
        "candidate_id": "C2:T0:1", "well": "C2", "timepoint": "T0", "x_px": 25.0, "y_px": 30.0,
        "diameter_px": 12.0, "integrated_label": "single", "integrated_confidence": 0.91,
        "integrated_round_id": "round-1", "is_duplicate_suppressed": False,
        "is_hierarchy_suppressed": False,
    }]).to_csv(artifact / "predictions" / "latest_integrated_predictions.csv", index=False)
    (artifact / "gated" / "plate_overview.json").write_text(json.dumps({
        "wells": [{"well": "C2", "final_category": "single_cell_origin", "final_category_label": "单细胞起源"}]
    }), encoding="utf-8")
    initialize_database(artifact / "annotations" / "annotations.db")
    manifest = tmp_path / "project.json"
    manifest.write_text(json.dumps({
        "project_id": "project-1", "project_name": "离线测试",
        "plates": [{
            "slug": "board-1", "board_id": "Board 1", "artifact_root": str(artifact),
            "images_manifest": str(artifact / "manifests" / "images.csv"),
            "report_json": str(artifact / "gated" / "plate_overview.json"),
        }],
    }), encoding="utf-8")
    return manifest, {"task_id": "task-1", "name": "离线审核测试", "status": "completed"}


def test_offline_bundle_is_self_contained_and_uses_relative_images(tmp_path: Path):
    manifest, task = _fixture(tmp_path)
    bundle, summary = build_offline_review_bundle(manifest, task, max_image_size=64)
    try:
        with ZipFile(bundle) as archive:
            names = set(archive.namelist())
            assert {"index.html", "data.js", "assets/offline-review.js", "assets/offline-review.css", "README.txt"} <= names
            assert "images/board-1/C2/T0.jpg" in names
            data_script = archive.read("data.js").decode("utf-8")
            assert BUNDLE_FORMAT in data_script
            assert str(tmp_path) not in data_script
            assert "images/board-1/C2/T0.jpg" in data_script
        assert summary == {
            "plate_count": 1, "image_count": 1, "object_count": 1,
            "missing_image_count": 0, "missing_images": [],
        }
    finally:
        bundle.unlink(missing_ok=True)


def test_offline_import_validates_identity_and_saves_decisions(tmp_path: Path):
    manifest, task = _fixture(tmp_path)
    payload = {
        "format": BUNDLE_FORMAT, "version": 1, "task_id": "task-1", "project_id": "project-1",
        "reviewer": "Reviewer", "plates": [{
            "slug": "board-1", "round_id": "round-1",
            "objects": [{"candidate_id": "C2:T0:1", "reviewed_label": "touching_doublet", "is_new": False}],
            "wells": [{"well": "C2", "screening_decision": "approved", "completed": True}],
        }],
    }
    result = import_offline_review_results(manifest, task, payload)
    assert result["updated_objects"] == 1
    assert result["updated_wells"] == 1

    artifact = Path(json.loads(manifest.read_text(encoding="utf-8"))["plates"][0]["artifact_root"])
    with sqlite3.connect(artifact / "annotations" / "annotations.db") as connection:
        assert connection.execute(
            "SELECT reviewed_label FROM integrated_training_reviews WHERE candidate_id='C2:T0:1'"
        ).fetchone()[0] == "touching_doublet"
        assert connection.execute(
            "SELECT decision FROM well_screening_reviews WHERE well='C2'"
        ).fetchone()[0] == "approved"

    bad = dict(payload, task_id="another-task")
    try:
        import_offline_review_results(manifest, task, bad)
    except ValueError as exc:
        assert "任务不匹配" in str(exc)
    else:
        raise AssertionError("mismatched task must be rejected")

    bad = dict(payload, project_id="another-project")
    try:
        import_offline_review_results(manifest, task, bad)
    except ValueError as exc:
        assert "项目不匹配" in str(exc)
    else:
        raise AssertionError("mismatched project must be rejected")


def test_nonempty_later_import_replaces_the_plate_review_state(tmp_path: Path):
    manifest, task = _fixture(tmp_path)
    payload = {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "task_id": "task-1",
        "project_id": "project-1",
        "reviewer": "First reviewer",
        "plates": [{
            "slug": "board-1",
            "round_id": "round-1",
            "objects": [{
                "candidate_id": "C2:T0:1",
                "reviewed_label": "touching_doublet",
                "is_new": False,
            }],
            "wells": [{
                "well": "C2",
                "screening_decision": "approved",
                "completed": True,
            }],
        }],
    }
    import_offline_review_results(manifest, task, payload)

    artifact = Path(json.loads(manifest.read_text(encoding="utf-8"))["plates"][0]["artifact_root"])
    database = artifact / "annotations" / "annotations.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO integrated_training_reviews "
            "(round_id, candidate_id, predicted_label, reviewed_label, decision, reviewer, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("round-1", "stale-candidate", "single", "debris", "corrected", "Old reviewer", "old"),
        )
        connection.execute(
            "INSERT INTO well_screening_reviews(well, decision, reviewer, notes, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("D3", "rejected", "Old reviewer", "old", "old"),
        )

    payload["reviewer"] = "Second reviewer"
    payload["plates"][0]["objects"][0]["reviewed_label"] = "debris"
    payload["plates"][0]["wells"][0]["screening_decision"] = "rejected"
    import_offline_review_results(manifest, task, payload)

    with sqlite3.connect(database) as connection:
        object_rows = connection.execute(
            "SELECT candidate_id, reviewed_label, reviewer "
            "FROM integrated_training_reviews WHERE round_id = 'round-1'"
        ).fetchall()
        well_rows = connection.execute(
            "SELECT well, decision, reviewer FROM well_screening_reviews"
        ).fetchall()
    assert object_rows == [("C2:T0:1", "debris", "Second reviewer")]
    assert well_rows == [("C2", "rejected", "Second reviewer")]


def test_empty_later_plate_preserves_existing_partial_review(tmp_path: Path):
    manifest, task = _fixture(tmp_path)
    reviewed_payload = {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "task_id": "task-1",
        "project_id": "project-1",
        "reviewer": "Production reviewer",
        "plates": [{
            "slug": "board-1",
            "round_id": "round-1",
            "objects": [{
                "candidate_id": "C2:T0:1",
                "reviewed_label": "touching_doublet",
                "is_new": False,
            }],
            "wells": [{
                "well": "C2",
                "screening_decision": "approved",
                "completed": True,
            }],
        }],
    }
    import_offline_review_results(manifest, task, reviewed_payload)

    empty_payload = {
        **reviewed_payload,
        "reviewer": "Offline reviewer",
        "plates": [{
            "slug": "board-1",
            "round_id": "round-1",
            "objects": [],
            "wells": [{
                "well": "C2",
                "screening_decision": "pending",
                "completed": False,
            }],
        }],
    }
    result = import_offline_review_results(manifest, task, empty_payload)

    artifact = Path(json.loads(manifest.read_text(encoding="utf-8"))["plates"][0]["artifact_root"])
    with sqlite3.connect(artifact / "annotations" / "annotations.db") as connection:
        object_row = connection.execute(
            "SELECT reviewed_label, reviewer FROM integrated_training_reviews "
            "WHERE round_id = 'round-1' AND candidate_id = 'C2:T0:1'"
        ).fetchone()
        well_row = connection.execute(
            "SELECT decision, reviewer FROM well_screening_reviews WHERE well = 'C2'"
        ).fetchone()
    assert object_row == ("touching_doublet", "Production reviewer")
    assert well_row == ("approved", "Production reviewer")
    assert result["preserved_empty_plates"] == 1


def test_offline_import_rebuilds_the_changed_plate_summary(tmp_path: Path, monkeypatch):
    import cellvision.offline_review as offline_review_module
    import cellvision.review_server as review_server_module

    manifest, task = _fixture(tmp_path)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["plates"][0]["config"] = "plate.yaml"
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    fake_config = {"paths": {}, "gated_report": {}}
    rebuilt: list[dict] = []
    monkeypatch.setattr(offline_review_module, "load_config", lambda _path: fake_config)
    monkeypatch.setattr(offline_review_module, "build_well_screening", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        review_server_module,
        "rebuild_quick_review_summary",
        lambda config: rebuilt.append(config) or {"status": "ready"},
    )
    payload = {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "task_id": "task-1",
        "project_id": "project-1",
        "reviewer": "Reviewer",
        "plates": [{
            "slug": "board-1",
            "round_id": "round-1",
            "objects": [{
                "candidate_id": "C2:T0:1",
                "reviewed_label": "single",
                "is_new": False,
            }],
            "wells": [{
                "well": "C2",
                "screening_decision": "approved",
                "completed": True,
            }],
        }],
    }

    result = import_offline_review_results(manifest, task, payload)

    assert rebuilt == [fake_config]
    assert result["refreshed_plates"] == 1
    assert result["refresh_warnings"] == []


def test_normal_review_results_can_be_exported_for_production_import(tmp_path: Path):
    manifest, task = _fixture(tmp_path)
    import_offline_review_results(manifest, task, {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "task_id": "task-1",
        "project_id": "project-1",
        "reviewer": "Reviewer",
        "plates": [{
            "slug": "board-1",
            "round_id": "round-1",
            "objects": [{
                "candidate_id": "C2:T0:1",
                "reviewed_label": "touching_doublet",
                "is_new": False,
            }],
            "wells": [{
                "well": "C2",
                "screening_decision": "approved",
                "completed": True,
            }],
        }],
    })

    exported = export_offline_review_results(manifest, task)

    assert exported["format"] == BUNDLE_FORMAT
    assert exported["task_id"] == "task-1"
    assert exported["project_id"] == "project-1"
    plate = exported["plates"][0]
    reviewed = next(item for item in plate["objects"] if item["candidate_id"] == "C2:T0:1")
    assert reviewed["reviewed_label"] == "touching_doublet"
    well = next(item for item in plate["wells"] if item["well"] == "C2")
    assert well == {"well": "C2", "screening_decision": "approved", "completed": True}


def test_offline_result_accepts_an_older_export_of_the_same_task_and_project(tmp_path: Path):
    manifest, task = _fixture(tmp_path)
    task["offline_export"] = {
        "summary": {
            "format": "cellvision-review-data",
            "package_id": "latest-package",
            "content_sha256": "content-1",
        }
    }
    payload = {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "task_id": "task-1",
        "project_id": "project-1",
        "reviewer": "Reviewer",
        "data_package": {
            "format": "cellvision-review-data",
            "package_id": "older-package",
            "content_sha256": "older-content",
        },
        "plates": [],
    }
    result = import_offline_review_results(manifest, task, payload)

    assert result["updated_objects"] == 0
    assert result["updated_wells"] == 0

    payload.pop("data_package")
    result = import_offline_review_results(manifest, task, payload)
    assert result["updated_objects"] == 0
