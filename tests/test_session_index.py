from pathlib import Path

from cellvision.session_index import parse_sessions_index, summarize_session_groups


def _write_session(root: Path, relative: str, *, wells: int = 96) -> None:
    folder = root.joinpath(*relative.replace("\\", "/").split("/"))
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, wells + 1):
        row = "ABCDEFGH"[(index - 1) // 12]
        column = (index - 1) % 12 + 1
        (folder / f"{row}{column}.tif").write_bytes(b"")


def test_sessions_idx_is_sorted_within_group_and_assigns_timepoints(tmp_path):
    _write_session(tmp_path, r"2026\06\23\first")
    _write_session(tmp_path, r"2026\07\07\last")
    xml = """<?xml version="1.0"?>
    <SessionList>
      <Sessions>
        <SessionHeader>
          <SessionID>late</SessionID>
          <Date>2026-07-07T15:14:07+08:00</Date>
          <ReceptacleIdentifier>QL2603 T6-2</ReceptacleIdentifier>
          <SessionFolder>2026\\07\\07\\last</SessionFolder>
        </SessionHeader>
        <SessionHeader>
          <SessionID>early</SessionID>
          <Date>2026-06-23T11:41:49+08:00</Date>
          <ReceptacleIdentifier>QL2603 T6-2</ReceptacleIdentifier>
          <SessionFolder>2026\\06\\23\\first</SessionFolder>
        </SessionHeader>
      </Sessions>
    </SessionList>"""
    index = tmp_path / "sessions.idx"
    index.write_text(xml, encoding="utf-8")

    result = parse_sessions_index(index, tmp_path, timepoint_origin=0)

    assert result["session_id"].tolist() == ["early", "late"]
    assert result["timepoint_label"].tolist() == ["T0", "T1"]
    assert result["zero_based_timepoint_label"].tolist() == ["T0", "T1"]
    assert result["day_label"].tolist() == ["Day0", "Day14"]
    assert result["group_id"].nunique() == 1
    assert result["group_complete"].tolist() == [True, True]


def test_session_group_summary_reports_missing_folder(tmp_path):
    xml = """<SessionList><Sessions>
      <SessionHeader><SessionID>x</SessionID><Date>2026-06-23T00:00:00+08:00</Date>
      <ReceptacleIdentifier>QL2603 T1-1</ReceptacleIdentifier><SessionFolder>missing</SessionFolder></SessionHeader>
    </Sessions></SessionList>"""
    index = tmp_path / "sessions.idx"
    index.write_text(xml, encoding="utf-8")

    sessions = parse_sessions_index(index, tmp_path)
    summary = summarize_session_groups(sessions)

    assert len(summary) == 1
    assert bool(summary.iloc[0]["folders_exist"]) is False
    assert int(summary.iloc[0]["complete_96_well_sessions"]) == 0
