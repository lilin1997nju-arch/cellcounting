"""Build a three-board, raw-data-only QL2603 production acceptance package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


DEFAULT_BOARDS = ("T1-2", "T2-1", "T4-2")
EXPECTED_BASELINE = {
    "T1-2": {"single_cell_origin": 40, "no_obvious_growth": 36, "multi_cell_origin": 14, "undetermined": 5, "positive_control": 1},
    "T2-1": {"single_cell_origin": 17, "no_obvious_growth": 67, "multi_cell_origin": 9, "undetermined": 2, "positive_control": 1},
    "T4-2": {"single_cell_origin": 31, "no_obvious_growth": 49, "multi_cell_origin": 11, "undetermined": 4, "positive_control": 1},
}


def _hash_copy(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        while chunk := incoming.read(1024 * 1024):
            outgoing.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    shutil.copystat(source, destination)
    return size, digest.hexdigest()


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as incoming:
        while chunk := incoming.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _acceptance_plan(
    boards: tuple[str, ...],
    commit: str,
    *,
    missing_board: str | None = None,
    missing_day: int | None = None,
) -> str:
    table = "\n".join(
        f"| {board} | {EXPECTED_BASELINE.get(board, {}).get('single_cell_origin', '—')} | "
        f"{EXPECTED_BASELINE.get(board, {}).get('no_obvious_growth', '—')} | "
        f"{EXPECTED_BASELINE.get(board, {}).get('multi_cell_origin', '—')} | "
        f"{EXPECTED_BASELINE.get(board, {}).get('undetermined', '—')} |"
        for board in boards
    )
    incomplete_note = ""
    expected_result = "2. 确认解析结果恰好为 3 块板、15 个 session；时间点为 T0、T1、T2、倒数第二次、末次。"
    compute_result = "3. 开始计算，确认三块板分别完成；记录 CPU/GPU 类型、每板耗时及任何告警。"
    if missing_board is not None and missing_day is not None:
        included_boards = [board for board in boards if board != missing_board]
        incomplete_note = f"""
## 时间点交集专项预期

- 原始目录可识别 3 块板、14 个 session。
- `QL2603 {missing_board}` 故意缺少 `Day{missing_day}`，该板已有的每个 session 仍是完整 96 孔。
- 使用默认时间点 Day0、Day1、Day2、Day14 创建任务时，应显示：满足条件 2 块、排除 1 块。
- 只允许 {"、".join(f"`QL2603 {board}`" for board in included_boards)} 进入计算；缺时间点的板不能导致任务创建失败或计算卡住。
"""
        expected_result = (
            f"2. 确认解析结果为 3 块板、14 个 session，并明确提示 `QL2603 {missing_board}` "
            f"缺少 Day{missing_day}；默认选择下满足条件 2 块、排除 1 块。"
        )
        compute_result = (
            "3. 开始计算，确认仅两块完整板进入队列并分别完成；缺时间点的板不得创建计算项，"
            "同时记录 CPU/GPU 类型、每板耗时及任何告警。"
        )
    return f"""# QL2603 三板生产验收包

构建 Git 节点：`{commit}`

本包只包含原始仪器数据，不包含 CellVision 既有计算产物或审核数据库。选择：

- T1-2：与新旧多重性准确率对比使用过的板，细胞起源分布较均衡；
- T2-1：无明显生长孔占比高，用于覆盖末点先筛除和较短计算路径；
- T4-2：与多重性复审及 C2 时序边界案例相关，用于覆盖时序改判与人工复核。
{incomplete_note}

## 完整人工验收步骤

1. 在生产首页选择本目录（含 `sessions.idx`），创建名为“QL2603 三板生产验收”的任务。
{expected_result}
{compute_result}
4. 进入每块板的审核页：T0～T2 可改判/补漏；倒数第二次和末次必须显示完整孔，默认 100%，不自动定位。
5. 检查页面没有训练、生成下一轮、单/双细胞专项训练审核或 Mask 审核入口。
6. 至少各审核 5 个孔；T4-2 必须包含 C2。保存后刷新页面，确认对象和孔结论仍存在。
7. 导出项目 Excel，确认“T0总细胞数、T1推测细胞数、T2推测细胞数”三列相邻。
8. 从已完成任务导出离线审核 ZIP，在另一台不运行 CellVision 的电脑解压并双击 `index.html`；完成 1 个孔后导出 JSON，再回生产任务导入。
9. 再次导出 Excel，确认离线审核结论已经回写。

## 历史结果仅作烟雾对照

模型与规则更新后允许变化，不要求逐孔复现；若板级分布发生大幅偏移，应暂停发布并复核。

| 板子 | 单细胞起源 | 无明显生长 | 多细胞来源 | 待定 |
|---|---:|---:|---:|---:|
{table}
"""


def build_package(
    source_root: Path,
    output_root: Path,
    boards: tuple[str, ...] = DEFAULT_BOARDS,
    *,
    repository: Path,
    make_zip: bool = True,
    missing_board: str | None = None,
    missing_day: int | None = None,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    source_index = source_root / "sessions.idx"
    if not source_index.is_file():
        raise FileNotFoundError(source_index)
    requested = {f"QL2603 {board}" for board in boards}
    tree = ET.parse(source_index)
    root = tree.getroot()
    sessions = root.find("Sessions")
    if sessions is None:
        raise ValueError("sessions.idx has no Sessions element")
    selected_headers = [
        header
        for header in list(sessions)
        if (header.findtext("ReceptacleIdentifier") or "").strip() in requested
    ]
    found = {(header.findtext("ReceptacleIdentifier") or "").strip() for header in selected_headers}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"boards not found in sessions.idx: {', '.join(missing)}")
    counts = {name: 0 for name in requested}
    for header in selected_headers:
        counts[(header.findtext("ReceptacleIdentifier") or "").strip()] += 1
    bad_counts = {name: count for name, count in counts.items() if count != 5}
    if bad_counts:
        raise ValueError(f"each board must have five sessions: {bad_counts}")

    if (missing_board is None) != (missing_day is None):
        raise ValueError("missing_board and missing_day must be provided together")
    omitted_session = None
    if missing_board is not None and missing_day is not None:
        target_name = f"QL2603 {missing_board}"
        if target_name not in requested:
            raise ValueError(f"missing board is not selected: {missing_board}")
        target_headers = [
            header
            for header in selected_headers
            if (header.findtext("ReceptacleIdentifier") or "").strip() == target_name
        ]
        acquisition_dates: list[tuple[ET.Element, date]] = []
        for header in target_headers:
            raw_date = (header.findtext("Date") or "").strip()
            try:
                acquisition_dates.append((header, date.fromisoformat(raw_date[:10])))
            except ValueError as exc:
                raise ValueError(f"invalid acquisition date for {target_name}: {raw_date}") from exc
        first_date = min(value for _, value in acquisition_dates)
        matches = [
            header
            for header, value in acquisition_dates
            if (value - first_date).days == int(missing_day)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {target_name} Day{missing_day} session, found {len(matches)}"
            )
        omitted_session = {
            "board": missing_board,
            "group_id": target_name,
            "day_label": f"Day{missing_day}",
            "session_id": (matches[0].findtext("SessionID") or "").strip(),
            "session_folder": (matches[0].findtext("SessionFolder") or "").strip(),
        }
        selected_headers.remove(matches[0])

    output_root.mkdir(parents=True)
    for header in list(sessions):
        sessions.remove(header)
    for header in selected_headers:
        sessions.append(header)
    tree.write(output_root / "sessions.idx", encoding="utf-8", xml_declaration=True)

    hashes: list[tuple[str, str]] = []
    total_bytes = 0
    file_count = 0
    for header in selected_headers:
        relative_text = (header.findtext("SessionFolder") or "").strip()
        relative = Path(relative_text.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe session path: {relative_text}")
        source_folder = (source_root / relative).resolve()
        if not source_folder.is_dir() or source_root not in source_folder.parents:
            raise FileNotFoundError(source_folder)
        for source in sorted(source_folder.rglob("*")):
            if not source.is_file():
                continue
            destination = output_root / relative / source.relative_to(source_folder)
            size, digest = _hash_copy(source, destination)
            rel = destination.relative_to(output_root).as_posix()
            hashes.append((rel, digest))
            total_bytes += size
            file_count += 1

    tags_source = source_root / "tags.csv"
    if tags_source.is_file():
        with tags_source.open("r", encoding="utf-8-sig", newline="") as incoming, (output_root / "tags.csv").open("w", encoding="utf-8", newline="") as outgoing:
            reader = csv.reader(incoming)
            writer = csv.writer(outgoing)
            for row in reader:
                if row and row[0].strip() in requested:
                    writer.writerow(row)
    for batch in source_root.glob("*.batchid"):
        shutil.copy2(batch, output_root / batch.name)

    commit = _git_commit(repository)
    (output_root / "ACCEPTANCE_PLAN.md").write_text(
        _acceptance_plan(
            boards,
            commit,
            missing_board=missing_board,
            missing_day=missing_day,
        ),
        encoding="utf-8",
    )
    metadata = {
        "format": "cellvision-raw-acceptance-package",
        "version": 1,
        "git_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_project": "QL2603",
        "boards": list(boards),
        "session_count": len(selected_headers),
        "raw_file_count": file_count,
        "raw_bytes": total_bytes,
        "contains_existing_cellvision_results": False,
        "intentionally_incomplete_session": omitted_session,
        "expected_historical_distribution": {board: EXPECTED_BASELINE.get(board, {}) for board in boards},
    }
    (output_root / "PACKAGE.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for small in [output_root / "sessions.idx", output_root / "tags.csv", output_root / "ACCEPTANCE_PLAN.md", output_root / "PACKAGE.json"]:
        if small.is_file():
            _, digest = _hash_file(small)
            hashes.append((small.relative_to(output_root).as_posix(), digest))
    for batch in output_root.glob("*.batchid"):
        _, digest = _hash_file(batch)
        hashes.append((batch.relative_to(output_root).as_posix(), digest))
    (output_root / "MANIFEST.sha256").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(hashes)),
        encoding="utf-8",
    )

    zip_path = output_root.with_suffix(".zip")
    if make_zip:
        if zip_path.exists():
            raise FileExistsError(zip_path)
        with ZipFile(zip_path, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for path in sorted(output_root.rglob("*")):
                if path.is_file():
                    archive.write(path, (Path(output_root.name) / path.relative_to(output_root)).as_posix())
    return {**metadata, "output": str(output_root), "zip": str(zip_path) if make_zip else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", nargs="+", default=list(DEFAULT_BOARDS))
    parser.add_argument("--missing-board")
    parser.add_argument("--missing-day", type=int)
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    result = build_package(
        args.source,
        args.output,
        tuple(args.boards),
        repository=repository,
        make_zip=not args.no_zip,
        missing_board=args.missing_board,
        missing_day=args.missing_day,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
