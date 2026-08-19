import json
from pathlib import Path

import pandas as pd
import yaml

from cellvision.project_images import materialize_project_images


def _write_session(root: Path, timepoint: str, wells: list[str], *, early: bool) -> Path:
    session = root / timepoint
    session.mkdir(parents=True)
    for well in wells:
        (session / f"{well}.tif").write_bytes(f"{timepoint}-{well}-raw".encode())
        if early:
            (session / f"{well}-cf.tif").write_bytes(f"{timepoint}-{well}-cf".encode())
            (session / f"{well}-cells.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    if early:
        (session / "metricsummary.csv").write_text("Well,Cell Confluence\nA2,1\n", encoding="utf-8")
    return session


def test_materialize_project_images_keeps_reviewable_wells_and_scrubs_no_growth(tmp_path: Path):
    source = tmp_path / "acquisition"
    sessions = {
        timepoint: _write_session(
            source,
            timepoint,
            ["A1", "A2", "A3"],
            early=timepoint in {"T0", "T1", "T2"},
        )
        for timepoint in ["T0", "T1", "T2", "T3", "T4"]
    }
    endpoint_csv = tmp_path / "endpoint.csv"
    pd.DataFrame([
        {"group_id": "Plate 1", "well": "A1", "is_positive_control": True, "raw_image_path": str(sessions["T4"] / "A1.tif"), "cf_mask_path": "source-cf"},
        {"group_id": "Plate 1", "well": "A2", "is_positive_control": False, "raw_image_path": str(sessions["T4"] / "A2.tif"), "cf_mask_path": "source-cf"},
        {"group_id": "Plate 1", "well": "A3", "is_positive_control": False, "raw_image_path": str(sessions["T4"] / "A3.tif"), "cf_mask_path": "source-cf"},
    ]).to_csv(endpoint_csv, index=False)

    project = tmp_path / "project"
    config_path = project / "configs" / "plate.yaml"
    config_path.parent.mkdir(parents=True)
    raw_config = {
        "paths": {"data_root": str(source), "artifact_root": str(project / "plates" / "plate")},
        "experiment": {"plate_id": "plate", "timepoint_directories": {tp: str(sessions[tp]) for tp in ["T0", "T1", "T2"]}},
        "review": {"late_timepoint_directories": {tp: str(sessions[tp]) for tp in ["T3", "T4"]}},
        "gated_report": {"day14_csv": str(endpoint_csv), "endpoint_csv": str(endpoint_csv)},
        "project_images": {"enabled": True, "project_root": str(project), "image_root": str(project / "data" / "images" / "plate"), "board_slug": "plate"},
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    result = materialize_project_images(
        config_path,
        raw_config,
        positive_wells={"A2"},
        gate_rows=[
            {"well": "A1", "is_positive_control": True},
            {"well": "A2", "is_positive_control": False},
            {"well": "A3", "is_positive_control": False},
        ],
        endpoint_csv=endpoint_csv,
        group_id="Plate 1",
    )

    image_root = project / "data" / "images" / "plate"
    for timepoint in ["T0", "T1", "T2"]:
        assert (image_root / timepoint / "A2.tif").is_file()
        assert (image_root / timepoint / "A2-cf.tif").is_file()
        assert not (image_root / timepoint / "A3.tif").exists()
    assert (image_root / "T3" / "A2.tif").is_file()
    assert (image_root / "T4" / "A2.tif").is_file()
    assert (image_root / "T4" / "A1.tif").is_file()
    assert not (image_root / "T4" / "A3.tif").exists()

    localized = pd.read_csv(result["endpoint_csv"], keep_default_na=False)
    paths = dict(zip(localized["well"], localized["raw_image_path"]))
    assert paths["A1"].startswith(str(project))
    assert paths["A2"].startswith(str(project))
    assert paths["A3"] == ""
    assert set(localized["cf_mask_path"]) == {""}

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert Path(saved["paths"]["data_root"]) == project / "data"
    assert all(str(project) in value for value in saved["experiment"]["timepoint_directories"].values())
    summary = json.loads((project / "data" / "manifests" / "plate-summary.json").read_text(encoding="utf-8"))
    assert summary["reviewable_wells"] == ["A2"]
    assert summary["no_growth_wells_without_images"] == ["A3"]
