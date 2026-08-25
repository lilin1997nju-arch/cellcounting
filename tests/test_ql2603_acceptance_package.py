import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from scripts.build_ql2603_acceptance_package import build_package


def test_acceptance_package_contains_only_selected_raw_sessions(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    root = ET.Element("SessionList")
    ET.SubElement(root, "Id").text = "batch"
    sessions = ET.SubElement(root, "Sessions")
    selected = ("T1-2", "T2-1", "T4-2")
    for board in (*selected, "T9-9"):
        for index in range(5):
            relative = Path("2026") / board / str(index)
            folder = source / relative
            folder.mkdir(parents=True)
            (folder / "A1.tif").write_bytes(f"{board}-{index}".encode())
            header = ET.SubElement(sessions, "SessionHeader")
            ET.SubElement(header, "SessionID").text = f"{board}-{index}"
            ET.SubElement(header, "ReceptacleIdentifier").text = f"QL2603 {board}"
            ET.SubElement(header, "SessionFolder").text = str(relative).replace("/", "\\")
    ET.ElementTree(root).write(source / "sessions.idx", encoding="utf-8", xml_declaration=True)
    with (source / "tags.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows([[f"QL2603 {board}", "C2", "2"] for board in (*selected, "T9-9")])
    (source / "batch.batchid").write_bytes(b"")

    output = tmp_path / "QL2603-acceptance"
    result = build_package(source, output, selected, repository=Path(__file__).parents[1])

    parsed = ET.parse(output / "sessions.idx")
    names = [item.findtext("ReceptacleIdentifier") for item in parsed.getroot().find("Sessions")]
    assert len(names) == 15
    assert set(names) == {f"QL2603 {board}" for board in selected}
    assert not any("T9-9" in str(path) for path in output.rglob("*"))
    assert result["raw_file_count"] == 15
    assert json.loads((output / "PACKAGE.json").read_text(encoding="utf-8"))["contains_existing_cellvision_results"] is False
    assert "T4-2 必须包含 C2" in (output / "ACCEPTANCE_PLAN.md").read_text(encoding="utf-8")

    manifest = {}
    for line in (output / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    assert manifest["2026/T4-2/4/A1.tif"] == hashlib.sha256(b"T4-2-4").hexdigest()
    with ZipFile(output.with_suffix(".zip")) as archive:
        assert f"{output.name}/sessions.idx" in archive.namelist()
        assert not any("T9-9" in name for name in archive.namelist())


def test_acceptance_package_can_omit_one_board_timepoint(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    root = ET.Element("SessionList")
    sessions = ET.SubElement(root, "Sessions")
    boards = ("T1-2", "T2-1", "T4-2")
    dates = ("2026-06-23", "2026-06-24", "2026-06-25", "2026-06-30", "2026-07-07")
    for board in boards:
        for index, acquisition_date in enumerate(dates):
            relative = Path("2026") / board / str(index)
            folder = source / relative
            folder.mkdir(parents=True)
            (folder / "A1.tif").write_bytes(f"{board}-{index}".encode())
            header = ET.SubElement(sessions, "SessionHeader")
            ET.SubElement(header, "SessionID").text = f"{board}-{index}"
            ET.SubElement(header, "Date").text = f"{acquisition_date}T08:00:00+08:00"
            ET.SubElement(header, "ReceptacleIdentifier").text = f"QL2603 {board}"
            ET.SubElement(header, "SessionFolder").text = str(relative).replace("/", "\\")
    ET.ElementTree(root).write(source / "sessions.idx", encoding="utf-8", xml_declaration=True)

    output = tmp_path / "intersection-test"
    result = build_package(
        source,
        output,
        boards,
        repository=Path(__file__).parents[1],
        make_zip=False,
        missing_board="T4-2",
        missing_day=1,
    )

    parsed = ET.parse(output / "sessions.idx").getroot().find("Sessions")
    assert parsed is not None
    selected = [item for item in parsed if item.findtext("ReceptacleIdentifier") == "QL2603 T4-2"]
    assert len(list(parsed)) == 14
    assert len(selected) == 4
    assert all(item.findtext("Date") != "2026-06-24T08:00:00+08:00" for item in selected)
    assert not (output / "2026" / "T4-2" / "1").exists()
    assert result["session_count"] == 14
    metadata = json.loads((output / "PACKAGE.json").read_text(encoding="utf-8"))
    assert metadata["intentionally_incomplete_session"]["day_label"] == "Day1"
    plan = (output / "ACCEPTANCE_PLAN.md").read_text(encoding="utf-8")
    assert "满足条件 2 块、排除 1 块" in plan
