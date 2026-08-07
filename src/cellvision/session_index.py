"""Parse vendor-exported ``sessions.idx`` files into grouped sessions.

The export format stores each acquisition as a ``SessionHeader``.  A
``ReceptacleIdentifier`` identifies the same plate/group across dates while
``SessionFolder`` points to the folder containing that acquisition's 96-well
images.  The XML order is not used for timepoints; sessions are ordered by
their acquisition timestamp within each receptacle.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_BOARD_RE = re.compile(r"^(?P<experiment>.+?)\s+(?P<board>T\d+[-_]\d+)$", re.IGNORECASE)
_WELL_TIF_RE = re.compile(r"^[A-H](?:[1-9]|1[0-2])\.tif$", re.IGNORECASE)
_CF_TIF_RE = re.compile(r"^[A-H](?:[1-9]|1[0-2])-cf\.tif$", re.IGNORECASE)


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _child_text(node: ET.Element, name: str, default: str = "") -> str:
    for child in node:
        if _local_name(child.tag) == name:
            return str(child.text or default).strip()
    return default


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _path_from_export(root: Path | None, session_folder: str) -> tuple[str, str]:
    """Return a normalized relative folder and an absolute path if possible."""

    raw = str(session_folder or "").strip()
    parts = [part for part in re.split(r"[\\/]+", raw) if part and part != "."]
    relative = Path(*parts) if parts else Path()
    candidate = Path(raw)
    if candidate.is_absolute():
        absolute = candidate
    elif root is not None:
        absolute = root / relative
    else:
        absolute = relative
    return relative.as_posix() if parts else "", str(absolute.resolve()) if parts else ""


def _folder_counts(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {
            "folder_exists": False,
            "well_image_count": 0,
            "cf_image_count": 0,
            "cells_csv_count": 0,
            "folder_file_count": 0,
        }
    files = [item for item in path.iterdir() if item.is_file()]
    return {
        "folder_exists": True,
        "well_image_count": sum(bool(_WELL_TIF_RE.match(item.name)) for item in files),
        "cf_image_count": sum(bool(_CF_TIF_RE.match(item.name)) for item in files),
        "cells_csv_count": sum(item.name.casefold().endswith("-cells.csv") for item in files),
        "folder_file_count": len(files),
    }


def _group_parts(identifier: str) -> tuple[str, str]:
    match = _BOARD_RE.match(identifier.strip())
    if not match:
        return identifier.strip(), ""
    return match.group("experiment").strip(), match.group("board").replace("_", "-").upper()


def parse_sessions_index(
    index_path: str | Path,
    data_root: str | Path | None = None,
    *,
    timepoint_origin: int = 0,
) -> pd.DataFrame:
    """Parse and group an exported ``sessions.idx`` file.

    ``timepoint_origin=0`` produces the model-facing labels ``T0`` ... ``Tn``.
    Set it to ``1`` for a one-based external convention.  Both the sequential
    timepoint and the actual calendar offset from the first session are
    emitted, so a group can also be addressed as ``Day0``, ``Day1`` ...
    """

    if int(timepoint_origin) < 0:
        raise ValueError("timepoint_origin must be non-negative")
    index = Path(index_path).expanduser().resolve()
    if not index.exists():
        raise FileNotFoundError(index)
    root = Path(data_root).expanduser().resolve() if data_root is not None else index.parent
    xml_root = ET.parse(index).getroot()
    headers = [node for node in xml_root.iter() if _local_name(node.tag) == "SessionHeader"]
    if not headers:
        raise ValueError(f"No SessionHeader entries found in {index}")

    records: list[dict[str, Any]] = []
    for source_index, header in enumerate(headers):
        identifier = _child_text(header, "ReceptacleIdentifier")
        session_folder = _child_text(header, "SessionFolder")
        relative, absolute = _path_from_export(root, session_folder)
        parsed_date = _parse_datetime(_child_text(header, "Date"))
        experiment, board = _group_parts(identifier)
        folder_path = Path(absolute) if absolute else Path()
        records.append(
            {
                "source_index": source_index,
                "session_id": _child_text(header, "SessionID"),
                "group_id": identifier,
                "experiment_group": experiment,
                "board_id": board,
                "date_raw": _child_text(header, "Date"),
                "acquisition_datetime": parsed_date.isoformat() if parsed_date else "",
                "acquisition_date": parsed_date.date().isoformat() if parsed_date else "",
                "session_folder_raw": session_folder,
                "session_folder_rel": relative,
                "session_path": absolute,
                "session_type": _child_text(header, "SessionTypeDescription"),
                "scan_session_type": _child_text(header, "ScanSessionType"),
                "excluded": _child_text(header, "Excluded").casefold() == "true",
                "awaiting_images": _child_text(header, "AwaitingImages").casefold() == "true",
                **_folder_counts(folder_path),
            }
        )

    frame = pd.DataFrame(records)
    sort_values = pd.to_datetime(frame["acquisition_datetime"], errors="coerce", utc=True)
    frame["_sort_datetime"] = sort_values
    frame = frame.sort_values(["group_id", "_sort_datetime", "source_index"], na_position="last").reset_index(drop=True)
    frame["session_ordinal"] = frame.groupby("group_id", sort=False).cumcount() + 1
    frame["timepoint_number"] = frame["session_ordinal"] - 1 + int(timepoint_origin)
    frame["timepoint_label"] = "T" + frame["timepoint_number"].astype(str)
    frame["zero_based_timepoint_label"] = "T" + (frame["session_ordinal"] - 1).astype(str)
    first_dates = frame.groupby("group_id")["acquisition_date"].transform("min")
    parsed_dates = pd.to_datetime(frame["acquisition_date"], errors="coerce")
    first_parsed_dates = pd.to_datetime(first_dates, errors="coerce")
    frame["culture_day"] = (parsed_dates - first_parsed_dates).dt.days.astype("Int64")
    frame["day_label"] = "Day" + frame["culture_day"].astype(str)
    frame["group_session_count"] = frame.groupby("group_id")["group_id"].transform("size")
    frame["group_complete"] = frame["folder_exists"] & frame["well_image_count"].ge(96)
    return frame.drop(columns=["_sort_datetime"])


def summarize_session_groups(sessions: pd.DataFrame) -> pd.DataFrame:
    """Return one row per plate/group for quick import validation."""

    if sessions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group_id, group in sessions.groupby("group_id", sort=True):
        group = group.sort_values("session_ordinal")
        rows.append(
            {
                "group_id": group_id,
                "experiment_group": str(group.iloc[0]["experiment_group"]),
                "board_id": str(group.iloc[0]["board_id"]),
                "session_count": int(len(group)),
                "first_acquisition": str(group.iloc[0]["acquisition_datetime"]),
                "last_acquisition": str(group.iloc[-1]["acquisition_datetime"]),
                "timepoints": ",".join(group["timepoint_label"].astype(str)),
                "folders_exist": bool(group["folder_exists"].all()),
                "complete_96_well_sessions": int(group["group_complete"].sum()),
                "missing_session_folders": int((~group["folder_exists"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_session_group_manifest(
    index_path: str | Path,
    data_root: str | Path | None,
    output_path: str | Path,
    *,
    timepoint_origin: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Write session-level and group-level CSVs plus a JSON summary."""

    sessions = parse_sessions_index(index_path, data_root, timepoint_origin=timepoint_origin)
    groups = summarize_session_groups(sessions)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output, index=False, encoding="utf-8-sig")
    group_output = output.with_name(f"{output.stem}.groups.csv")
    groups.to_csv(group_output, index=False, encoding="utf-8-sig")
    summary = {
        "index_path": str(Path(index_path).expanduser().resolve()),
        "data_root": str(Path(data_root).expanduser().resolve()) if data_root else "",
        "timepoint_origin": int(timepoint_origin),
        "session_count": int(len(sessions)),
        "group_count": int(len(groups)),
        "groups_with_missing_folders": int((~groups["folders_exist"]).sum()) if not groups.empty else 0,
        "groups_with_complete_96_well_sessions": int(groups["complete_96_well_sessions"].eq(groups["session_count"]).sum()) if not groups.empty else 0,
        "session_folder_count": int(sessions["folder_exists"].sum()),
        "session_96_well_count": int(sessions["group_complete"].sum()),
        "group_ids": groups["group_id"].astype(str).tolist() if not groups.empty else [],
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return sessions, groups, summary_path
