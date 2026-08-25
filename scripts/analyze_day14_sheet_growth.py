from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from cellvision.session_index import parse_sessions_index


WELL_NAMES = [f"{row}{column}" for row in "ABCDEFGH" for column in range(1, 13)]


def _component_metrics(
    mask_path: Path,
    confluence_pct: float,
    *,
    downsample: int,
    minimum_component_coverage_pct: float,
    minimum_radius_px: float,
    minimum_mean_distance_px: float,
) -> dict[str, float | int | bool]:
    with Image.open(mask_path) as opened:
        width = max(1, opened.width // downsample)
        height = max(1, opened.height // downsample)
        mask = np.asarray(
            opened.resize((width, height), Image.Resampling.NEAREST), dtype=bool
        )

    foreground_pixels = int(mask.sum())
    if foreground_pixels == 0 or confluence_pct <= 0:
        return {
            "component_count": 0,
            "largest_component_coverage_pct": 0.0,
            "largest_sheet_coverage_pct": 0.0,
            "sheet_coverage_pct": 0.0,
            "maximum_component_radius_px": 0.0,
            "maximum_sheet_radius_px": 0.0,
            "sheet_component_count": 0,
            "has_sheet_component": False,
        }

    # The vendor confluence is foreground / effective well area.  Recovering
    # that denominator avoids counting image corners outside the circular well.
    effective_well_area = foreground_pixels / max(confluence_pct / 100.0, 1e-9)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:]
    objects = ndimage.find_objects(labels)
    largest_coverage = 0.0
    maximum_radius = 0.0
    sheet_coverages: list[float] = []
    sheet_radii: list[float] = []

    # Only the largest foreground components can form an obvious sheet.  This
    # bound keeps the scan fast while retaining far more components than the
    # final decision can use.
    candidate_indices = np.argsort(areas)[-min(len(areas), 32) :][::-1]
    for index in candidate_indices:
        coverage = float(areas[index] / effective_well_area * 100.0)
        largest_coverage = max(largest_coverage, coverage)
        if coverage < minimum_component_coverage_pct:
            continue
        component = labels[objects[index]] == index + 1
        distance = ndimage.distance_transform_edt(component)
        radius = float(distance.max() * downsample)
        mean_distance = float(distance[component].mean() * downsample)
        maximum_radius = max(maximum_radius, radius)
        # Thin, long wall/rim responses can cover several percent of a well but
        # have little interior thickness.  True confluent cell sheets have both
        # substantial area and a thick interior.
        if radius >= minimum_radius_px and mean_distance >= minimum_mean_distance_px:
            sheet_coverages.append(coverage)
            sheet_radii.append(radius)

    return {
        "component_count": int(count),
        "largest_component_coverage_pct": largest_coverage,
        "largest_sheet_coverage_pct": max(sheet_coverages, default=0.0),
        "sheet_coverage_pct": float(sum(sheet_coverages)),
        "maximum_component_radius_px": maximum_radius,
        "maximum_sheet_radius_px": max(sheet_radii, default=0.0),
        "sheet_component_count": len(sheet_coverages),
        "has_sheet_component": bool(sheet_coverages),
    }


def analyze(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    sessions = parse_sessions_index(args.index, args.root)
    requested_group_ids = {
        str(value) for value in getattr(args, "group_id", []) if str(value).strip()
    }
    if requested_group_ids:
        available_group_ids = set(sessions["group_id"].astype(str))
        missing_group_ids = sorted(requested_group_ids - available_group_ids)
        if missing_group_ids:
            raise RuntimeError(
                "Requested groups were not found: " + ", ".join(missing_group_ids)
            )
        sessions = sessions[
            sessions["group_id"].astype(str).isin(requested_group_ids)
        ].copy()
    available_days = pd.to_numeric(sessions["culture_day"], errors="coerce").dropna().astype(int)
    endpoint_day = int(args.endpoint_day) if str(args.endpoint_day).strip() else (
        max(day for day in available_days.unique() if int(day) >= 7)
        if any(int(day) >= 7 for day in available_days.unique())
        else -1
    )
    if endpoint_day < 0:
        raise RuntimeError("No culture day >= 7 was found")
    endpoint_label = f"Day{endpoint_day}"
    endpoint = sessions.loc[sessions["day_label"].eq(endpoint_label)].copy()
    if endpoint.empty:
        raise RuntimeError(f"No {endpoint_label} sessions were found")

    rows: list[dict[str, object]] = []
    for session in endpoint.itertuples(index=False):
        folder = Path(str(session.session_path))
        metrics_path = folder / "metricsummary.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path).set_index("Well")
        for well in WELL_NAMES:
            if well not in metrics.index:
                continue
            confluence = float(metrics.loc[well, "Cell Confluence"])
            count = int(metrics.loc[well, "Cell Count"])
            mask_path = folder / f"{well}-cf.tif"
            component = {
                "component_count": 0,
                "largest_component_coverage_pct": 0.0,
                "largest_sheet_coverage_pct": 0.0,
                "sheet_coverage_pct": 0.0,
                "maximum_component_radius_px": 0.0,
                "maximum_sheet_radius_px": 0.0,
                "sheet_component_count": 0,
                "has_sheet_component": False,
            }
            if confluence >= args.minimum_confluence_to_analyze and mask_path.exists():
                component = _component_metrics(
                    mask_path,
                    confluence,
                    downsample=args.downsample,
                    minimum_component_coverage_pct=args.minimum_component_coverage,
                    minimum_radius_px=args.minimum_radius,
                    minimum_mean_distance_px=args.minimum_mean_distance,
                )
            strict_positive = bool(component["has_sheet_component"]) or confluence >= args.high_confluence_rescue
            rows.append(
                {
                    "group_id": session.group_id,
                    "board_id": session.board_id,
                    "day_label": session.day_label,
                    "acquisition_date": session.acquisition_date,
                    "well": well,
                    "is_positive_control": well == "A1",
                    "cell_count": count,
                    "instrument_confluence_pct": confluence,
                    **component,
                    "day14_obvious_sheet_growth": strict_positive,
                    "endpoint_day_label": endpoint_label,
                    "endpoint_obvious_sheet_growth": strict_positive,
                    "screening_decision": "retain" if strict_positive else "exclude",
                    "raw_image_path": str(folder / f"{well}.tif"),
                    "cf_mask_path": str(mask_path),
                }
            )

    wells = pd.DataFrame(rows)
    biological = wells.loc[~wells["is_positive_control"]].copy()
    summaries: list[dict[str, object]] = []
    for group_id, group in biological.groupby("group_id", sort=True):
        positive = group["day14_obvious_sheet_growth"].astype(bool)
        summaries.append(
            {
                "group_id": group_id,
                "board_id": str(group.iloc[0]["board_id"]),
                "well_count_excluding_A1": int(len(group)),
                "retained_obvious_sheet_wells": int(positive.sum()),
                "excluded_non_sheet_wells": int((~positive).sum()),
                "retained_fraction_pct": float(positive.mean() * 100.0),
                "median_instrument_confluence_pct": float(group["instrument_confluence_pct"].median()),
                "maximum_instrument_confluence_pct": float(group["instrument_confluence_pct"].max()),
                "median_sheet_coverage_pct": float(group["sheet_coverage_pct"].median()),
                "maximum_sheet_coverage_pct": float(group["sheet_coverage_pct"].max()),
            }
        )
    plates = pd.DataFrame(summaries)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = "day14" if endpoint_day == 14 else f"endpoint_day{endpoint_day}"
    wells_path = output / f"ql2603_{suffix}_well_screening.csv"
    plates_path = output / f"ql2603_{suffix}_plate_summary.csv"
    summary_path = output / f"ql2603_{suffix}_screening_summary.json"
    wells.to_csv(wells_path, index=False, encoding="utf-8-sig")
    plates.to_csv(plates_path, index=False, encoding="utf-8-sig")
    payload = {
        "dataset": "QL2603",
        "endpoint_day_label": endpoint_label,
        "endpoint_session_count": int(endpoint["group_id"].nunique()),
        "day14_session_count": int(endpoint["group_id"].nunique()) if endpoint_day == 14 else 0,
        "well_count_excluding_A1": int(len(biological)),
        "retained_obvious_sheet_wells": int(biological["day14_obvious_sheet_growth"].sum()),
        "excluded_non_sheet_wells": int((~biological["day14_obvious_sheet_growth"].astype(bool)).sum()),
        "thresholds": {
            "minimum_confluence_to_analyze_pct": args.minimum_confluence_to_analyze,
            "minimum_component_coverage_pct": args.minimum_component_coverage,
            "minimum_component_radius_px": args.minimum_radius,
            "minimum_component_mean_distance_px": args.minimum_mean_distance,
            "high_confluence_rescue_pct": args.high_confluence_rescue,
            "endpoint_day": endpoint_day,
        },
        "positive_rule": "thick sheet component OR high total confluence rescue",
        "A1_excluded_as_positive_control": True,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return wells_path, plates_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Day14 obvious-sheet growth screening")
    parser.add_argument("--root", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--output", default="artifacts/day14_screening/ql2603")
    parser.add_argument("--endpoint-day", default="", help="Actual culture day used as the endpoint; blank selects the latest available day >= 7")
    parser.add_argument(
        "--group-id",
        action="append",
        default=[],
        help="Only analyze this group ID; repeat to include multiple boards",
    )
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--minimum-confluence-to-analyze", type=float, default=3.0)
    # A smaller 0.5% pilot threshold retained sparse edge colonies such as
    # QL2603 T5-2/G1.  One percent better matches the requested definition of
    # visibly sheet-like Day14 growth.
    parser.add_argument("--minimum-component-coverage", type=float, default=1.0)
    parser.add_argument("--minimum-radius", type=float, default=40.0)
    parser.add_argument("--minimum-mean-distance", type=float, default=10.0)
    parser.add_argument("--high-confluence-rescue", type=float, default=20.0)
    args = parser.parse_args()
    if not args.index:
        args.index = str(Path(args.root) / "sessions.idx")
    for path in analyze(args):
        print(path)


if __name__ == "__main__":
    main()
