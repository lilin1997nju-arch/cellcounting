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
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from PIL import Image, ImageOps

from .config import PROJECT_ROOT, artifact_path, load_config
from .gated_screening import build_gated_plate_report
from .multiplicity import ensure_integrated_review_table, save_integrated_reviews
from .review_helpers import _visible_v2_review_instances, _with_final_decisions
from .review_storage import initialize_database, save_annotation
from .review_summary import latest_prediction_path
from .well_screening import build_well_screening, ensure_well_screening_review_table


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
    try:
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(assets / "offline-review.html", "index.html")
            archive.write(assets / "offline-review.css", "assets/offline-review.css")
            archive.write(assets / "offline-review.js", "assets/offline-review.js")
            for index, plate in enumerate(manifest.get("plates", []), start=1):
                if not isinstance(plate, dict):
                    continue
                paths = _plate_paths(plate, manifest_file)
                if not paths["images_manifest"].is_file():
                    raise FileNotFoundError(paths["images_manifest"])
                initialize_database(paths["database"])
                images = pd.read_csv(paths["images_manifest"])
                images["well"] = images["well"].astype(str).str.upper()
                images["timepoint"] = images["timepoint"].astype(str).str.upper()
                round_id, objects = _review_objects(paths["artifact_root"], paths["database"])
                report = _read_json(paths["report"]) if paths["report"].is_file() else {}
                report_lookup = {
                    str(row.get("well") or "").upper(): row
                    for row in report.get("wells", [])
                    if isinstance(row, dict)
                }
                screening_reviews: dict[str, str] = {}
                with sqlite3.connect(paths["database"]) as connection:
                    ensure_well_screening_review_table(paths["database"])
                    try:
                        screening_reviews = {
                            str(row[0]).upper(): str(row[1])
                            for row in connection.execute("SELECT well, decision FROM well_screening_reviews")
                        }
                    except sqlite3.OperationalError:
                        pass
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
                            continue
                        member = f"images/{slug}/{well}/{timepoint}.jpg"
                        jpeg = zip_path.parent / f".{zip_path.stem}-{index}-{well}-{timepoint}.jpg"
                        try:
                            width, height = _write_jpeg(source, jpeg, max_image_size, jpeg_quality)
                            archive.write(jpeg, member)
                        finally:
                            jpeg.unlink(missing_ok=True)
                        image_count += 1
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
                        "screening_decision": screening_reviews.get(well, "pending"),
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
    added_objects = 0
    skipped_incomplete_wells = 0
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
        source = latest_prediction_path(paths["artifact_root"])
        if source is None:
            raise ValueError(f"板子 {slug} 没有可用预测结果")
        predictions = pd.read_csv(source, low_memory=False)
        prediction_lookup = predictions.set_index(predictions["candidate_id"].astype(str), drop=False)
        round_id = str(predictions.iloc[0].get("integrated_round_id") or result_plate.get("round_id") or "offline-import")
        completed_wells = {
            str(item.get("well") or "").upper()
            for item in result_plate.get("wells", [])
            if isinstance(item, dict) and bool(item.get("completed"))
        }
        skipped_incomplete_wells += sum(
            1 for item in result_plate.get("wells", [])
            if isinstance(item, dict) and not bool(item.get("completed"))
        )
        review_items: list[dict[str, Any]] = []
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
                image_row = selected.iloc[0]
                temporary_id = candidate_id or f"offline:{well}:{timepoint}:{added_objects + 1}"
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
        if review_items:
            updated_objects += save_integrated_reviews(database, round_id, review_items, reviewer)
        ensure_well_screening_review_table(database)
        with sqlite3.connect(database) as connection:
            for well_item in result_plate.get("wells", []):
                if not isinstance(well_item, dict):
                    continue
                if not bool(well_item.get("completed")):
                    continue
                decision = str(well_item.get("screening_decision") or "pending")
                if decision not in {"approved", "rejected", "pending"}:
                    raise ValueError(f"孔结论状态无效：{decision}")
                connection.execute(
                    "INSERT INTO well_screening_reviews(well, decision, reviewer, notes, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(well) DO UPDATE SET "
                    "decision=excluded.decision, reviewer=excluded.reviewer, notes=excluded.notes, updated_at=excluded.updated_at",
                    (str(well_item.get("well") or "").upper(), decision, reviewer, "offline_review_import", datetime.now(timezone.utc).isoformat()),
                )
                updated_wells += 1
        config_value = str(plate.get("config") or "")
        if completed_wells and config_value:
            try:
                config = load_config(_resolve(config_value, relative_to=manifest_file.parent))
                build_well_screening(config, database, selected_wells=completed_wells)
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
                refreshed_plates += 1
            except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
                refresh_warnings.append(f"{slug}: {exc}")
    return {
        "status": "imported",
        "updated_objects": updated_objects,
        "added_objects": added_objects,
        "updated_wells": updated_wells,
        "skipped_incomplete_wells": skipped_incomplete_wells,
        "refreshed_plates": refreshed_plates,
        "refresh_warnings": refresh_warnings,
        "reviewer": reviewer,
    }
