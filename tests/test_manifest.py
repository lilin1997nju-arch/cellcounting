from pathlib import Path

import pandas as pd
import pytest

from cellvision.manifest import _find_session, all_wells, normalize_well, stable_sequence_id


def test_well_parsing():
    assert normalize_well("h06") == "H6"
    assert normalize_well("A12") == "A12"
    with pytest.raises(ValueError):
        normalize_well("I1")


def test_plate_has_96_unique_wells():
    wells = all_wells()
    assert len(wells) == 96
    assert len(set(wells)) == 96


def test_sequence_id_is_stable():
    first = stable_sequence_id("exp", "plate", "B3")
    second = stable_sequence_id("exp", "plate", "b03")
    assert first == second


def test_filtered_session_without_a1_is_still_discovered(tmp_path: Path):
    session = tmp_path / "T0"
    session.mkdir()
    (session / "A2.tif").write_bytes(b"raw")
    (session / "A2-cf.tif").write_bytes(b"mask")

    assert _find_session(session) == session


def test_existing_split_has_no_well_leakage():
    split_dir = Path("artifacts/splits")
    if not split_dir.exists():
        pytest.skip("split not generated yet")
    sets = []
    for name in ("train", "validation", "test"):
        sets.append(set(pd.read_csv(split_dir / f"{name}_sequences.csv")["well"]))
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
