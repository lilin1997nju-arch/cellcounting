"""Portable, server-free review bundles for completed project tasks.

The bundle deliberately uses plain HTML/CSS/JavaScript and relative JPEG
paths.  It can therefore be opened from ``file://`` on an offline Windows PC
without Python, a web server, or a browser extension.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from PIL import Image, ImageOps

from .config import PROJECT_ROOT, artifact_path, load_config
from .gated_screening import build_gated_plate_report
from .multiplicity import ensure_integrated_review_table, save_integrated_reviews
from .review_helpers import _visible_v2_review_instances, _with_final_decisions
from .review_storage import initialize_database, save_annotation
from .review_summary import latest_prediction_path
from .well_screening import (
    build_well_screening,
    ensure_well_screening_review_table,
    ensure_well_timepoint_cell_count_review_table,
)
from .review_data_package import DATA_PACKAGE_FORMAT, DATA_PACKAGE_MANIFEST


BUNDLE_FORMAT = "cellvision-offline-review"
BUNDLE_VERSION = 1
LABELS = {
    "single",
    "touching_doublet",
    "cluster_3plus",
    "debris",
    "invalid",
    "uncertain",
}


def _review_data_identity(manifest_file: Path) -> dict[str, Any] | None:
    package_root = manifest_file.parent.parent
    metadata_path = package_root / DATA_PACKAGE_MANIFEST
    if not metadata_path.is_file():
        return None
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if metadata.get("format") != DATA_PACKAGE_FORMAT:
        return None
    return {
        "format": DATA_PACKAGE_FORMAT,
        "package_id": str(metadata.get("package_id") or ""),
        "content_sha256": str(metadata.get("content_sha256") or ""),
        "production_git_commit": str(metadata.get("production_git_commit") or ""),
    }


def _resolve(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((relative_to or PROJECT_ROOT) / path).resolve()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _plate_paths(plate: dict[str, Any], manifest_path: Path) -> dict[str, Path]:
    artifact_root = _resolve(
        plate.get("artifact_root") or "", relative_to=manifest_path.parent
    )
    images_manifest = _resolve(
        plate.get("images_manifest") or artifact_root / "manifests" / "images.csv",
        relative_to=manifest_path.parent,
    )
    report = _resolve(
        plate.get("report_json")
        or plate.get("gated_output_dir", "") and Path(str(plate["gated_output_dir"])) / "plate_overview.json"
        or artifact_root / "gated" / "plate_overview.json",
        relative_to=manifest_path.parent,
    )
    return {
        "artifact_root": artifact_root,
        "images_manifest": images_manifest,
        "report": report,
        "database": artifact_root / "annotations" / "annotations.db",
    }


def _review_objects(artifact_root: Path, database: Path) -> tuple[str, pd.DataFrame]:
    source = latest_prediction_path(artifact_root)
    if source is None:
        return "", pd.DataFrame()
    frame = pd.read_csv(source, low_memory=False)
    if frame.empty:
        return "", frame
    for column, default in {
        "integrated_label": "uncertain",
        "integrated_confidence": 0.0,
        "integrated_round_id": "offline-unknown",
        "is_duplicate_suppressed": False,
        "is_hierarchy_suppressed": False,
        "v3_track_behavior": "",
    }.items():
        if column not in frame:
            frame[column] = default
    visible_target = frame["integrated_label"].astype(str).isin(LABELS - {"invalid"})
    wall_target = frame["v3_track_behavior"].astype(str).eq("wall_structure_invalid")
    frame = frame[(visible_target | wall_target) & frame["well"].astype(str).str.upper().ne("A1")].copy()
    if frame.empty:
        return "", frame
    round_id = str(frame.iloc[0]["integrated_round_id"])
    ensure_integrated_review_table(database)
    with sqlite3.connect(database) as connection:
        reviews = pd.read_sql_query(
            "SELECT candidate_id, reviewed_label, decision, updated_at, integrated_review_id "
            "FROM integrated_training_reviews ORDER BY updated_at, integrated_review_id",
            connection,
        )
        try:
            manual = pd.read_sql_query(
                "SELECT candidate_id, well, timepoint, x_px, y_px, diameter_px, "
                "reviewed_label, updated_at FROM quick_missed_objects",
                connection,
            )
        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            manual = pd.DataFrame()
    if not reviews.empty:
        reviews = reviews.drop_duplicates("candidate_id", keep="last")
        frame = frame.merge(reviews, on="candidate_id", how="left")
    else:
        frame["reviewed_label"] = None
        frame["decision"] = None
    frame["current_label"] = frame["reviewed_label"].fillna(frame["integrated_label"])
    frame["is_manual_missed"] = False
    if not manual.empty:
        manual["integrated_label"] = manual["reviewed_label"]
        manual["current_label"] = manual["reviewed_label"]
        manual["integrated_confidence"] = 1.0
        manual["integrated_round_id"] = round_id
        manual["is_manual_missed"] = True
        manual["decision"] = "approved"
        manual = manual[~manual["candidate_id"].astype(str).isin(frame["candidate_id"].astype(str))]
        frame = pd.concat([frame, manual], ignore_index=True, sort=False)
    if "v2_instance_id" in frame:
        frame = _visible_v2_review_instances(frame)
    else:
        frame = frame[
            ~frame["is_duplicate_suppressed"].fillna(False).astype(bool)
            & ~frame["is_hierarchy_suppressed"].fillna(False).astype(bool)
        ].copy()
    frame = _with_final_decisions(frame)
    return round_id, frame


def _write_jpeg(source: Path, destination: Path, max_size: int, quality: int) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=quality, optimize=True)
        return image.size


def build_offline_review_bundle(
    manifest_path: str | Path,
    task: dict[str, Any],
    *,
    ui_root: str | Path | None = None,
    max_image_size: int = 1400,
    jpeg_quality: int = 86,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create a ZIP and return its temporary path plus a compact summary."""

    manifest_file = _resolve(manifest_path)
    manifest = _read_json(manifest_file)
    assets = _resolve(ui_root or PROJECT_ROOT / "review-ui")
    temporary = NamedTemporaryFile(prefix="cellvision-offline-review-", suffix=".zip", delete=False)
    zip_path = Path(temporary.name)
    temporary.close()
    data: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "task_id": str(task.get("task_id") or ""),
            "name": str(task.get("name") or manifest.get("project_name") or "审核任务"),
            "created_by": str(task.get("created_by") or ""),
        },
        "project": {
            "project_id": str(manifest.get("project_id") or manifest_file.parent.name),
            "project_name": str(manifest.get("project_name") or manifest_file.parent.name),
        },
        "labels": sorted(LABELS),
        "plates": [],
    }
    image_count = 0
    object_count = 0
    missing_images: list[str] = []
    prepared_plates: list[
        tuple[int, dict[str, Any], dict[str, Path], pd.DataFrame]
    ] = []
    total_images = 0
    try:
        for index, plate in enumerate(manifest.get("plates", []), start=1):
            if not isinstance(plate, dict):
                continue
            paths = _plate_paths(plate, manifest_file)
            if not paths["images_manifest"].is_file():
                raise FileNotFoundError(paths["images_manifest"])
            images = pd.read_csv(paths["images_manifest"])
            images["well"] = images["well"].astype(str).str.upper()
            images["timepoint"] = images["timepoint"].astype(str).str.upper()
            if "decode_status" in images:
                total_images += int(images["decode_status"].astype(str).eq("ok").sum())
            else:
                total_images += int(len(images))
            prepared_plates.append((index, plate, paths, images))
        if progress_callback is not None:
            progress_callback(0, total_images, "已读取项目数据，开始打包审核图像")

        processed_images = 0
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(assets / "offline-review.html", "index.html")
            archive.write(assets / "offline-review.css", "assets/offline-review.css")
            archive.write(assets / "offline-review.js", "assets/offline-review.js")
            for index, plate, paths, images in prepared_plates:
                initialize_database(paths["database"])
                round_id, objects = _review_objects(paths["artifact_root"], paths["database"])
                report = _read_json(paths["report"]) if paths["report"].is_file() else {}
                report_lookup = {
                    str(row.get("well") or "").upper(): row
                    for row in report.get("wells", [])
                    if isinstance(row, dict)
                }
                screening_reviews: dict[str, str] = {}
                cell_count_overrides: dict[str, dict[str, int]] = {}
                ensure_well_screening_review_table(paths["database"])
                ensure_well_timepoint_cell_count_review_table(paths["database"])
                with sqlite3.connect(paths["database"]) as connection:
                    try:
                        screening_reviews = {
                            str(row[0]).upper(): str(row[1])
                            for row in connection.execute("SELECT well, decision FROM well_screening_reviews")
                        }
                    except sqlite3.OperationalError:
                        pass
                    for row_well, timepoint, cell_count in connection.execute(
                        "SELECT well, timepoint, cell_count "
                        "FROM well_timepoint_cell_count_reviews"
                    ):
                        cell_count_overrides.setdefault(
                            str(row_well).upper(), {}
                        )[str(timepoint).upper()] = int(cell_count)
                slug = str(plate.get("slug") or plate.get("board_id") or f"plate-{index}")
                plate_data: dict[str, Any] = {
                    "slug": slug,
                    "board_id": str(plate.get("board_id") or plate.get("group_id") or slug),
                    "round_id": round_id,
                    "wells": [],
                }
                all_wells = sorted(
                    set(images["well"]) | set(objects.get("well", pd.Series(dtype=str)).astype(str).str.upper()) | set(report_lookup),
                    key=lambda value: (ord(value[:1] or "Z") - ord("A"), int(value[1:]) if value[1:].isdigit() else 999),
                )
                for well in all_wells:
                    well_images: dict[str, Any] = {}
                    for row in images[images["well"].eq(well)].itertuples(index=False):
                        timepoint = str(row.timepoint).upper()
                        if str(getattr(row, "decode_status", "ok")) != "ok":
                            continue
                        source = _resolve(str(row.raw_image_path), relative_to=manifest_file.parent)
                        if not source.is_file():
                            missing_images.append(f"{slug}/{well}/{timepoint}")
                            processed_images += 1
                            if progress_callback is not None:
                                progress_callback(
                                    processed_images,
                                    total_images,
                                    f"正在打包 {plate_data['board_id']} · {well} {timepoint}",
                                )
                            continue
                        member = f"images/{slug}/{well}/{timepoint}.jpg"
                        jpeg = zip_path.parent / f".{zip_path.stem}-{index}-{well}-{timepoint}.jpg"
                        try:
                            width, height = _write_jpeg(source, jpeg, max_image_size, jpeg_quality)
                            archive.write(jpeg, member)
                        finally:
                            jpeg.unlink(missing_ok=True)
                        image_count += 1
                        processed_images += 1
                        if progress_callback is not None:
                            progress_callback(
                                processed_images,
                                total_images,
                                f"正在打包 {plate_data['board_id']} · {well} {timepoint}",
                            )
                        well_images[timepoint] = {
                            "url": member,
                            "width": width,
                            "height": height,
                            "source_width": int(getattr(row, "width_px", width)),
                            "source_height": int(getattr(row, "height_px", height)),
                            "annotatable": timepoint in {"T0", "T1", "T2"},
                        }
                    local = objects[objects["well"].astype(str).str.upper().eq(well)].copy() if not objects.empty else pd.DataFrame()
                    columns = [
                        "candidate_id", "timepoint", "x_px", "y_px", "diameter_px",
                        "integrated_label", "current_label", "final_label", "reviewed_label", "integrated_confidence",
                        "cell_probability", "debris_probability", "invalid_probability", "is_manual_missed",
                    ]
                    for column in columns:
                        if column not in local:
                            local[column] = None
                    object_rows = local[columns].to_dict(orient="records")
                    object_count += len(object_rows)
                    report_row = report_lookup.get(well, {})
                    plate_data["wells"].append({
                        "well": well,
                        "images": well_images,
                        "objects": _json_safe(object_rows),
                        "screening_decision": screening_reviews.get(well, "unclassified"),
                        "cell_count_overrides": cell_count_overrides.get(well, {}),
                        "report": _json_safe({
                            "final_category": report_row.get("final_category"),
                            "final_category_label": report_row.get("final_category_label"),
                            "undetermined_reason": report_row.get("undetermined_reason"),
                            "endpoint_obvious_growth": report_row.get("endpoint_obvious_growth", report_row.get("day14_obvious_growth")),
                        }),
                    })
                data["plates"].append(plate_data)
            payload = json.dumps(_json_safe(data), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
            archive.writestr("data.js", f"window.CELLVISION_OFFLINE_DATA={payload};\n")
            archive.writestr(
                "README.txt",
                "双击 index.html 开始离线审核。审核进度保存在当前浏览器。\r\n"
                "完成后点击“导出审核结果”，将 JSON 文件复制回生产电脑并在任务页导入。\r\n"
                "本包只含审核网页与降采样图像，不包含模型、训练功能或原始 TIFF。\r\n",
            )
        if progress_callback is not None:
            progress_callback(total_images, total_images, "离线审核包已生成")
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise
    return zip_path, {
        "plate_count": len(data["plates"]),
        "image_count": image_count,
        "object_count": object_count,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:20],
    }


def import_offline_review_results(
    manifest_path: str | Path,
    task: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate and apply an exported offline result document."""

    manifest_file = _resolve(manifest_path)
    manifest = _read_json(manifest_file)
    if payload.get("format") != BUNDLE_FORMAT or int(payload.get("version", 0)) != BUNDLE_VERSION:
        raise ValueError("不是受支持的 CellVision 离线审核结果")
    if str(payload.get("task_id") or "") != str(task.get("task_id") or ""):
        raise ValueError("审核结果与当前任务不匹配")
    if str(payload.get("project_id") or "") != str(manifest.get("project_id") or manifest_file.parent.name):
        raise ValueError("审核结果与当前项目不匹配")
    reviewer = str(payload.get("reviewer") or "offline_reviewer").strip() or "offline_reviewer"
    plate_lookup = {
        str(item.get("slug") or item.get("board_id") or ""): item
        for item in manifest.get("plates", []) if isinstance(item, dict)
    }
    updated_objects = 0
    updated_wells = 0
    updated_cell_count_overrides = 0
    added_objects = 0
    skipped_incomplete_wells = 0
    preserved_empty_plates = 0
    refreshed_plates = 0
    refresh_warnings: list[str] = []
    for result_plate in payload.get("plates", []):
        if not isinstance(result_plate, dict):
            continue
        slug = str(result_plate.get("slug") or "")
        plate = plate_lookup.get(slug)
        if plate is None:
            raise ValueError(f"审核结果包含未知板子：{slug}")
        paths = _plate_paths(plate, manifest_file)
        database = initialize_database(paths["database"])
        result_well_items = [
            item for item in result_plate.get("wells", []) if isinstance(item, dict)
        ]
        completed_wells = {
            str(item.get("well") or "").upper()
            for item in result_well_items
            if bool(item.get("completed"))
        }
        skipped_incomplete_wells += sum(
            1 for item in result_well_items if not bool(item.get("completed"))
        )
        cell_count_replacement_wells: set[str] = set()
        cell_count_items: list[tuple[str, str, int]] = []
        for well_item in result_well_items:
            if "cell_count_overrides" not in well_item:
                continue
            well = str(well_item.get("well") or "").upper()
            if not well:
                raise ValueError("人工细胞总数缺少孔号")
            overrides = well_item.get("cell_count_overrides")
            if overrides is None:
                overrides = {}
            if not isinstance(overrides, dict):
                raise ValueError(f"孔 {well} 的人工细胞总数格式无效")
            cell_count_replacement_wells.add(well)
            for raw_timepoint, raw_count in overrides.items():
                timepoint = str(raw_timepoint).upper()
                if timepoint not in {"T0", "T1", "T2"}:
                    raise ValueError(f"孔 {well} 的时间点无效：{timepoint}")
                if isinstance(raw_count, bool):
                    raise ValueError(f"孔 {well} {timepoint} 的人工细胞总数无效")
                try:
                    count = int(raw_count)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"孔 {well} {timepoint} 的人工细胞总数无效"
                    ) from exc
                if count < 0 or count > 10000 or count != raw_count:
                    raise ValueError(f"孔 {well} {timepoint} 的人工细胞总数无效")
                cell_count_items.append((well, timepoint, count))
        if not completed_wells and not cell_count_replacement_wells:
            ensure_integrated_review_table(database)
            ensure_well_screening_review_table(database)
            ensure_well_timepoint_cell_count_review_table(database)
            with sqlite3.connect(database) as connection:
                has_existing_reviews = any((
                    connection.execute(
                        "SELECT 1 FROM integrated_training_reviews LIMIT 1"
                    ).fetchone(),
                    connection.execute(
                        "SELECT 1 FROM quick_missed_objects LIMIT 1"
                    ).fetchone(),
                    connection.execute(
                        "SELECT 1 FROM well_screening_reviews LIMIT 1"
                    ).fetchone(),
                    connection.execute(
                        "SELECT 1 FROM well_timepoint_cell_count_reviews LIMIT 1"
                    ).fetchone(),
                ))
            if has_existing_reviews:
                preserved_empty_plates += 1
            continue
        source = latest_prediction_path(paths["artifact_root"])
        if source is None:
            raise ValueError(f"板子 {slug} 没有可用预测结果")
        predictions = pd.read_csv(source, low_memory=False)
        prediction_lookup = predictions.set_index(predictions["candidate_id"].astype(str), drop=False)
        round_id = str(predictions.iloc[0].get("integrated_round_id") or result_plate.get("round_id") or "offline-import")
        review_items: list[dict[str, Any]] = []
        manual_items: list[tuple[dict[str, Any], dict[str, Any], str]] = []
        completed_well_items: list[tuple[str, str]] = []
        for well_item in result_well_items:
            if not isinstance(well_item, dict) or not bool(well_item.get("completed")):
                continue
            decision = str(well_item.get("screening_decision") or "unclassified")
            if decision not in {"approved", "rejected", "pending", "unclassified"}:
                raise ValueError(f"孔结论状态无效：{decision}")
            completed_well_items.append(
                (str(well_item.get("well") or "").upper(), decision)
            )
        image_manifest = pd.read_csv(paths["images_manifest"])
        for item in result_plate.get("objects", []):
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id") or "")
            reviewed = str(item.get("reviewed_label") or "")
            if reviewed not in LABELS:
                raise ValueError(f"对象 {candidate_id} 的标签无效")
            if bool(item.get("is_new")):
                well = str(item.get("well") or "").upper()
                if well not in completed_wells:
                    continue
                timepoint = str(item.get("timepoint") or "").upper()
                selected = image_manifest[
                    image_manifest["well"].astype(str).str.upper().eq(well)
                    & image_manifest["timepoint"].astype(str).str.upper().eq(timepoint)
                ]
                if selected.empty:
                    raise ValueError(f"补漏对象对应图像不存在：{slug}/{well}/{timepoint}")
                manual_items.append((item, selected.iloc[0].to_dict(), reviewed))
                continue
            if candidate_id not in prediction_lookup.index:
                raise ValueError(f"板子 {slug} 中找不到对象：{candidate_id}")
            prediction_row = prediction_lookup.loc[candidate_id]
            if str(prediction_row.get("well") or "").upper() not in completed_wells:
                continue
            predicted = str(prediction_row.get("integrated_label") or "uncertain")
            if predicted not in LABELS:
                predicted = "invalid" if predicted == "invalid" else "uncertain"
            review_items.append({"candidate_id": candidate_id, "predicted_label": predicted, "reviewed_label": reviewed})

        ensure_integrated_review_table(database)
        ensure_well_screening_review_table(database)
        ensure_well_timepoint_cell_count_review_table(database)
        with sqlite3.connect(database) as connection:
            if completed_wells:
                annotation_ids = [
                    int(row[0])
                    for row in connection.execute(
                        "SELECT annotation_id FROM quick_missed_objects WHERE round_id = ?",
                        (round_id,),
                    ).fetchall()
                ]
                connection.execute(
                    "DELETE FROM quick_missed_objects WHERE round_id = ?", (round_id,)
                )
                if annotation_ids:
                    placeholders = ", ".join("?" for _ in annotation_ids)
                    connection.execute(
                        f"DELETE FROM annotations WHERE annotation_id IN ({placeholders})",
                        annotation_ids,
                    )
                connection.execute(
                    "DELETE FROM integrated_training_reviews WHERE round_id = ?",
                    (round_id,),
                )
                connection.execute("DELETE FROM well_screening_reviews")
            for well in cell_count_replacement_wells:
                connection.execute(
                    "DELETE FROM well_timepoint_cell_count_reviews WHERE well = ?",
                    (well,),
                )
            updated = datetime.now(timezone.utc).isoformat()
            for well, timepoint, count in cell_count_items:
                connection.execute(
                    "INSERT INTO well_timepoint_cell_count_reviews "
                    "(well, timepoint, cell_count, reviewer, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (well, timepoint, count, reviewer, updated),
                )
                updated_cell_count_overrides += 1

        if review_items:
            updated_objects += save_integrated_reviews(database, round_id, review_items, reviewer)
        for item, image_row, reviewed in manual_items:
            well = str(item.get("well") or "").upper()
            timepoint = str(item.get("timepoint") or "").upper()
            temporary_id = str(item.get("candidate_id") or "") or f"offline:{well}:{timepoint}:{added_objects + 1}"
            object_type = "cell" if reviewed in {"single", "touching_doublet", "cluster_3plus"} else "debris" if reviewed == "debris" else "irrelevant" if reviewed == "invalid" else "uncertain"
            annotation_id = save_annotation(database, {
                "sequence_id": str(image_row.get("experiment_id") or "offline"),
                "plate_id": str(image_row.get("plate_id") or slug),
                "well": well, "timepoint": timepoint,
                "object_id": temporary_id, "canonical_target_id": temporary_id,
                "track_id": temporary_id, "parent_track_id": None,
                "x_px": float(item.get("x_px", 0)), "y_px": float(item.get("y_px", 0)),
                "object_type": object_type,
                "viability": "unknown" if object_type == "cell" else "not_applicable",
                "division_state": "unknown", "duplicate_of": None,
                "reviewer": reviewer, "confidence": 1.0, "notes": "offline_review_missed",
            })
            candidate_id = f"{well}:{timepoint}:manual:{annotation_id}"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO quick_missed_objects "
                    "(round_id, annotation_id, candidate_id, well, timepoint, x_px, y_px, diameter_px, reviewed_label, reviewer, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (round_id, annotation_id, candidate_id, well, timepoint, float(item.get("x_px", 0)),
                     float(item.get("y_px", 0)), float(item.get("diameter_px", 18)), reviewed, reviewer,
                     datetime.now(timezone.utc).isoformat()),
                )
            added_objects += 1
        with sqlite3.connect(database) as connection:
            for well, decision in completed_well_items:
                connection.execute(
                    "INSERT INTO well_screening_reviews(well, decision, reviewer, notes, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(well) DO UPDATE SET "
                    "decision=excluded.decision, reviewer=excluded.reviewer, notes=excluded.notes, updated_at=excluded.updated_at",
                    (well, decision, reviewer, "offline_review_import", datetime.now(timezone.utc).isoformat()),
                )
                updated_wells += 1
        config_value = str(plate.get("config") or "")
        refreshed_wells = completed_wells | cell_count_replacement_wells
        if refreshed_wells and config_value:
            try:
                config = load_config(_resolve(config_value, relative_to=manifest_file.parent))
                build_well_screening(config, database, selected_wells=refreshed_wells)
                settings = config.get("gated_report", {})
                endpoint_csv = settings.get("endpoint_csv") or settings.get("day14_csv")
                if endpoint_csv and settings.get("group_id") and settings.get("output_dir"):
                    endpoint_timepoint = str(settings.get("endpoint_timepoint", "T4")).upper()
                    with sqlite3.connect(database) as connection:
                        try:
                            late_rows = connection.execute(
                                "SELECT well, decision FROM late_growth_reviews WHERE timepoint = ?",
                                (endpoint_timepoint,),
                            ).fetchall()
                        except sqlite3.OperationalError:
                            late_rows = []
                    build_gated_plate_report(
                        endpoint_csv,
                        settings["group_id"],
                        settings["output_dir"],
                        early_screening_csv=artifact_path(config, "predictions", "latest_well_screening.csv"),
                        sessions_csv=settings.get("sessions_csv"),
                        locate_day7=False,
                        day14_growth_overrides={str(well).upper(): str(decision) for well, decision in late_rows},
                        endpoint_day_label=str(settings.get("endpoint_day_label", "Day14")),
                    )
                from .review_server import rebuild_quick_review_summary

                rebuild_quick_review_summary(config)
                refreshed_plates += 1
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                sqlite3.Error,
                pd.errors.ParserError,
            ) as exc:
                refresh_warnings.append(f"{slug}: {exc}")
    return {
        "status": "imported",
        "updated_objects": updated_objects,
        "added_objects": added_objects,
        "updated_wells": updated_wells,
        "updated_cell_count_overrides": updated_cell_count_overrides,
        "skipped_incomplete_wells": skipped_incomplete_wells,
        "preserved_empty_plates": preserved_empty_plates,
        "refreshed_plates": refreshed_plates,
        "refresh_warnings": refresh_warnings,
        "reviewer": reviewer,
    }


def export_offline_review_results(
    manifest_path: str | Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Serialize reviews made with the normal review UI for production import."""

    manifest_file = _resolve(manifest_path)
    manifest = _read_json(manifest_file)
    payload: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "task_id": str(task.get("task_id") or manifest.get("task_id") or ""),
        "project_id": str(manifest.get("project_id") or manifest_file.parent.name),
        "reviewer": "offline_reviewer",
        "plates": [],
    }
    data_identity = _review_data_identity(manifest_file)
    if data_identity is not None:
        payload["data_package"] = data_identity
    reviewers: list[str] = []
    for plate in manifest.get("plates", []):
        if not isinstance(plate, dict):
            continue
        paths = _plate_paths(plate, manifest_file)
        database = initialize_database(paths["database"])
        round_id, objects = _review_objects(paths["artifact_root"], database)
        completed_wells: set[str] = set()
        screening: dict[str, str] = {}
        cell_count_overrides: dict[str, dict[str, int]] = {}
        ensure_well_timepoint_cell_count_review_table(database)
        with sqlite3.connect(database) as connection:
            try:
                rows = connection.execute(
                    "SELECT well, reviewer FROM quick_review_sessions ORDER BY updated_at"
                ).fetchall()
                completed_wells.update(str(row[0]).upper() for row in rows)
                reviewers.extend(str(row[1]).strip() for row in rows if row[1])
            except sqlite3.OperationalError:
                pass
            try:
                rows = connection.execute(
                    "SELECT well, decision, reviewer FROM well_screening_reviews"
                ).fetchall()
                for well, decision, reviewer in rows:
                    well = str(well).upper()
                    screening[well] = str(decision)
                    if str(decision) in {"approved", "pending", "rejected"}:
                        completed_wells.add(well)
                    if reviewer:
                        reviewers.append(str(reviewer).strip())
            except sqlite3.OperationalError:
                pass
            rows = connection.execute(
                "SELECT well, timepoint, cell_count, reviewer "
                "FROM well_timepoint_cell_count_reviews"
            ).fetchall()
            for well, timepoint, cell_count, reviewer in rows:
                well = str(well).upper()
                cell_count_overrides.setdefault(well, {})[str(timepoint).upper()] = int(cell_count)
                if reviewer:
                    reviewers.append(str(reviewer).strip())

        object_rows: list[dict[str, Any]] = []
        if not objects.empty:
            for row in objects.to_dict(orient="records"):
                well = str(row.get("well") or "").upper()
                candidate_id = str(row.get("candidate_id") or "")
                reviewed = str(
                    row.get("reviewed_label")
                    or row.get("final_label")
                    or row.get("current_label")
                    or row.get("integrated_label")
                    or "uncertain"
                )
                if reviewed not in LABELS:
                    reviewed = "uncertain"
                is_new = bool(row.get("is_manual_missed")) or ":manual:" in candidate_id
                value = {
                    "candidate_id": candidate_id,
                    "reviewed_label": reviewed,
                    "is_new": is_new,
                }
                if is_new:
                    value.update({
                        "well": well,
                        "timepoint": str(row.get("timepoint") or "").upper(),
                        "x_px": float(row.get("x_px") or 0),
                        "y_px": float(row.get("y_px") or 0),
                        "diameter_px": float(row.get("diameter_px") or 18),
                    })
                object_rows.append(value)

        all_wells = sorted(
            set(objects.get("well", pd.Series(dtype=str)).astype(str).str.upper())
            | set(screening)
            | set(cell_count_overrides)
            | completed_wells
        )
        payload["plates"].append({
            "slug": str(plate.get("slug") or plate.get("board_id") or ""),
            "round_id": round_id,
            "objects": object_rows,
            "wells": [
                {
                    "well": well,
                    "screening_decision": screening.get(well, "unclassified"),
                    "completed": well in completed_wells,
                    "cell_count_overrides": cell_count_overrides.get(well, {}),
                }
                for well in all_wells
            ],
        })
    if reviewers:
        payload["reviewer"] = next((value for value in reversed(reviewers) if value), "offline_reviewer")
    return _json_safe(payload)
