from __future__ import annotations

from pathlib import Path

import pytest

from cellvision.config import load_config


def test_nested_base_config_merge_preserves_siblings(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "paths:\n  data_root: /data\n  artifact_root: /artifacts\n"
        "runtime:\n  device: auto\n  mixed_precision: true\n",
        encoding="utf-8",
    )
    child.write_text(
        "base_config: base.yaml\nruntime:\n  device: cpu\n",
        encoding="utf-8",
    )
    config = load_config(child)
    assert config["runtime"] == {"device": "cpu", "mixed_precision": True, "host": "127.0.0.1", "port": 8777}


def test_environment_paths_override_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "config.yaml"
    path.write_text("paths:\n  data_root: /yaml-data\n  artifact_root: /yaml-artifacts\n", encoding="utf-8")
    monkeypatch.setenv("CELLVISION_DATA_ROOT", str(tmp_path / "mounted-data"))
    monkeypatch.setenv("CELLVISION_ARTIFACT_ROOT", str(tmp_path / "mounted-artifacts"))
    config = load_config(path)
    assert config["paths"]["data_root"] == str((tmp_path / "mounted-data").resolve())
    assert config["paths"]["artifact_root"] == str((tmp_path / "mounted-artifacts").resolve())


def test_missing_required_path_is_rejected(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text("paths:\n  artifact_root: /artifacts\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data_root"):
        load_config(path)

