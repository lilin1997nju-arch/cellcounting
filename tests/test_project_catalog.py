import json
import sqlite3
from pathlib import Path

from cellvision.project_catalog import ProjectCatalog


def _fixture_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    collection = tmp_path / "projects"
    project_dir = collection / "demo"
    artifact_root = project_dir / "artifacts"
    gated_root = artifact_root / "gated"
    gated_root.mkdir(parents=True)
    manifest = project_dir / "project.json"
    manifest.write_text(
        json.dumps(
            {
                "project_id": "demo",
                "project_name": "Demo",
                "created_by": "reviewer",
                "plates": [
                    {
                        "slug": "plate-1",
                        "status": "completed",
                        "artifact_root": str(artifact_root),
                        "gated_output_dir": str(gated_root),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return collection, manifest, gated_root / "plate_overview.json"


def _write_report(path: Path, first_category: str) -> None:
    path.write_text(
        json.dumps(
            {
                "well_count": 2,
                "category_counts": {
                    first_category: 1,
                    "no_obvious_growth": 1,
                },
                "wells": [
                    {
                        "well": "A1",
                        "final_category": first_category,
                        "undetermined_reason": "",
                    },
                    {
                        "well": "A2",
                        "final_category": "no_obvious_growth",
                        "undetermined_reason": "",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_catalog_refreshes_reports_and_preserves_human_well_decision(tmp_path: Path):
    collection, manifest, report = _fixture_project(tmp_path)
    _write_report(report, "single_cell_origin")
    catalog = ProjectCatalog(collection / "project_catalog.sqlite")

    first = catalog.sync_manifest(manifest, force=True)
    assert first is not None
    assert first["category_counts"]["single_cell_origin"] == 1

    catalog.sync_review_update(
        manifest,
        "demo:plate-1",
        wells={"A1"},
        source="quick_review",
        reviewer="alice",
        action_id="42",
    )
    _write_report(report, "multi_cell_origin")
    refreshed = catalog.sync_manifest(manifest, force=True)

    assert refreshed is not None
    assert refreshed["category_counts"]["single_cell_origin"] == 1
    assert refreshed["category_counts"].get("multi_cell_origin", 0) == 0
    with sqlite3.connect(catalog.path) as connection:
        current = connection.execute(
            "SELECT category_code, source, reviewer, model_category_code FROM well_current WHERE well='A1'"
        ).fetchone()
        history = connection.execute(
            "SELECT operation, source, reviewer FROM well_decision_history WHERE well='A1' ORDER BY decision_id"
        ).fetchall()
    assert current == ("single_cell_origin", "quick_review", "alice", "multi_cell_origin")
    assert history[-1] == ("review_save", "quick_review", "alice")


def test_catalog_task_progress_is_queryable(tmp_path: Path):
    collection, manifest, report = _fixture_project(tmp_path)
    _write_report(report, "single_cell_origin")
    catalog = ProjectCatalog(collection / "project_catalog.sqlite")
    catalog.sync_manifest(manifest, force=True)

    task = catalog.sync_task(
        {
            "task_id": "task-1",
            "project_id": "demo",
            "project_manifest": str(manifest),
            "name": "Demo task",
            "created_by": "alice",
            "status": "running",
            "progress_stage": "inference",
            "progress_current": 1,
            "progress_total": 2,
            "progress_percent": 50,
            "progress_boards": [
                {
                    "slug": "plate-1",
                    "status": "running",
                    "stage": "v2_temporal_evidence",
                    "progress_percent": 50,
                    "elapsed_seconds": 12.5,
                }
            ],
        }
    )
    assert task is not None
    assert task["progress_percent"] == 50
    with sqlite3.connect(catalog.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM task_plates").fetchone()[0] == 1
        plate = connection.execute(
            "SELECT status, current_stage, progress_percent, elapsed_seconds FROM plates WHERE plate_id='demo:plate-1'"
        ).fetchone()
    assert plate == ("running", "v2_temporal_evidence", 50.0, 12.5)


def test_catalog_hides_system_placeholder_and_reports_detection_dates(tmp_path: Path):
    collection = tmp_path / "projects"
    placeholder = collection / "active" / "project.json"
    placeholder.parent.mkdir(parents=True)
    placeholder.write_text(json.dumps({
        "project_id": "active",
        "project_name": "Cell Vision Production",
        "system_placeholder": True,
        "plates": [],
    }), encoding="utf-8")
    project = collection / "real" / "project.json"
    project.parent.mkdir(parents=True)
    project.write_text(json.dumps({
        "project_id": "real",
        "project_name": "Real project",
        "detection_start_date": "2026-06-23",
        "detection_end_date": "2026-07-07",
        "plates": [],
    }), encoding="utf-8")
    catalog = ProjectCatalog(collection / "project_catalog.sqlite")

    assert catalog.sync_manifest(placeholder, force=True) is None
    catalog.sync_manifest(project, force=True)
    result = catalog.list_projects()

    assert result["total"] == 1
    assert [item["project_id"] for item in result["items"]] == ["real"]
    assert result["items"][0]["detection_start_date"] == "2026-06-23"
    assert result["items"][0]["detection_end_date"] == "2026-07-07"
