import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from cellvision import review_data_package
from cellvision.config import load_config
from cellvision.review_data_package import (
    DATA_PACKAGE_FORMAT,
    prepare_review_data_package,
    validate_review_data_package,
)
from cellvision.review_platform import register_review_package


def _project(tmp_path: Path) -> tuple[Path, dict]:
    project = tmp_path / "project"
    image_dir = project / "data" / "images" / "board-1" / "T0"
    image_dir.mkdir(parents=True)
    Image.new("L", (24, 20), 128).save(image_dir / "C2.tif")
    Image.new("L", (24, 20), 255).save(image_dir / "C2-cf.tif")
    (project / "cache").mkdir()
    (project / "cache" / "render.jpg").write_bytes(b"cache")
    (project / "models").mkdir()
    (project / "models" / "model.pt").write_bytes(b"model")

    artifact = project / "plates" / "board-1"
    (artifact / "manifests").mkdir(parents=True)
    (artifact / "gated").mkdir()
    pd.DataFrame([{
        "experiment_id": "exp",
        "plate_id": "plate",
        "well": "C2",
        "timepoint": "T0",
        "raw_image_path": str(image_dir / "C2.tif"),
        "cf_image_path": str(image_dir / "C2-cf.tif"),
        "decode_status": "ok",
    }]).to_csv(artifact / "manifests" / "images.csv", index=False)
    (artifact / "gated" / "plate_overview.json").write_text(
        json.dumps({"category_counts": {"single_cell_origin": 1}}), encoding="utf-8"
    )
    config = project / "configs" / "board-1.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({
        "paths": {"data_root": str(project / "data"), "artifact_root": str(artifact)},
        "runtime": {"ignore_path_env_overrides": True},
        "experiment": {
            "experiment_id": "exp",
            "plate_id": "plate",
            "plate_rows": "ABCDEFGH",
            "plate_columns": 12,
            "timepoint_aliases": {},
            "timepoint_directories": {"T0": str(image_dir)},
        },
        "calibration": {"resolution_um_per_pixel": 2.08},
        "review": {"late_timepoint_directories": {}},
    }), encoding="utf-8")
    manifest = project / "project.json"
    manifest.write_text(json.dumps({
        "project_id": "project-1",
        "project_name": "Project 1",
        "task_id": "task-1",
        "root": str(project),
        "source": {"root": str(tmp_path / "acquisition"), "access_policy": "ingest_only"},
        "image_storage": {"mode": "project_owned_after_endpoint_gate", "root": str(project / "data" / "images")},
        "plates": [{
            "slug": "board-1",
            "board_id": "Board 1",
            "config": str(config),
            "artifact_root": str(artifact),
            "images_manifest": str(artifact / "manifests" / "images.csv"),
            "report_json": str(artifact / "gated" / "plate_overview.json"),
            "status": "completed",
        }],
    }), encoding="utf-8")
    (project / "task_queue.json").write_text(json.dumps([{
        "task_id": "task-1",
        "name": "Task 1",
        "status": "completed",
        "path": str(project),
        "source_path": str(tmp_path / "acquisition"),
        "project_manifest": str(manifest),
        "offline_export": {"status": "completed"},
    }]), encoding="utf-8")
    return manifest, {"task_id": "task-1", "name": "Task 1", "status": "completed"}


def test_review_data_package_is_relative_verified_and_environment_free(tmp_path: Path, monkeypatch):
    manifest, task = _project(tmp_path)
    monkeypatch.setattr(
        review_data_package,
        "_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("export must not hash files")),
    )
    package, summary = prepare_review_data_package(
        manifest, task, git_commit="abc123"
    )

    metadata = validate_review_data_package(package)
    assert package.suffix == ".cvreview"
    assert metadata["format"] == DATA_PACKAGE_FORMAT
    assert metadata["environment_included"] is False
    assert metadata["models_included"] is False
    assert metadata["integrity_mode"] == "size-and-presence"
    assert "content_sha256" not in metadata
    assert all("sha256" not in item for item in metadata["files"])
    assert summary["package_id"] == metadata["package_id"]
    reused_package, reused_summary = prepare_review_data_package(
        manifest, task, git_commit="abc123"
    )
    assert reused_package == package
    assert reused_summary["reused"] is True
    assert (package / "project" / "data" / "images" / "board-1" / "T0" / "C2.tif").is_file()
    assert not (package / "project" / "data" / "images" / "board-1" / "T0" / "C2-cf.tif").exists()
    assert not (package / "project" / "cache").exists()
    assert not (package / "project" / "models").exists()

    portable_manifest = json.loads((package / "project" / "project.json").read_text(encoding="utf-8"))
    assert portable_manifest["root"] == "."
    assert portable_manifest["plates"][0]["config"] == "configs/board-1.yaml"
    queue = json.loads((package / "project" / "task_queue.json").read_text(encoding="utf-8"))
    assert queue[0]["project_manifest"] == "project.json"
    assert "offline_export" not in queue[0]

    config = load_config(package / "project" / "configs" / "board-1.yaml")
    assert Path(config["paths"]["artifact_root"]) == package / "project" / "plates" / "board-1"
    assert Path(config["experiment"]["timepoint_directories"]["T0"]) == package / "project" / "data" / "images" / "board-1" / "T0"
    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in package.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".json", ".yaml", ".csv"}
    )
    assert str(tmp_path) not in all_text


def test_review_data_package_detects_tampering(tmp_path: Path):
    manifest, task = _project(tmp_path)
    package, _ = prepare_review_data_package(manifest, task, git_commit="abc123")
    image = package / "project" / "data" / "images" / "board-1" / "T0" / "C2.tif"
    image.write_bytes(b"tampered")
    try:
        validate_review_data_package(package)
    except ValueError as exc:
        assert "大小不符" in str(exc) or "校验失败" in str(exc)
    else:
        raise AssertionError("tampered package must be rejected")


def test_installed_review_platform_registers_package_and_tracks_identity(tmp_path: Path, monkeypatch):
    manifest, task = _project(tmp_path)
    package, _ = prepare_review_data_package(manifest, task, git_commit="abc123")
    monkeypatch.setenv("CELLVISION_REVIEW_HOME", str(tmp_path / "review-home"))
    image = package / "project" / "data" / "images" / "board-1" / "T0" / "C2.tif"
    changed = bytearray(image.read_bytes())
    changed[-1] ^= 1
    image.write_bytes(changed)

    entrypoint, metadata = register_review_package(package)

    assert entrypoint == package / "project" / "project.json"
    registry = json.loads((tmp_path / "review-home" / "registry.json").read_text(encoding="utf-8"))
    assert registry["packages"][0]["package_id"] == metadata["package_id"]
    assert registry["packages"][0]["package_path"] == str(package)
    assert registry["packages"][0]["hashes_verified"] is False

    from cellvision.project_server import create_project_app

    app = create_project_app(entrypoint)
    project_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", "") == "/api/project"
    )
    project = project_endpoint(project_id="project-1")
    assert project["project_id"] == "project-1"
    assert project["portable_review"] is True
