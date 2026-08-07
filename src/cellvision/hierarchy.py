from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
GROUP_LABELS = {"touching_doublet", "cluster_3plus"}


def _series(
    frame: pd.DataFrame, name: str, default: Any, dtype: Any | None = None
) -> pd.Series:
    value = frame.get(name, pd.Series(default, index=frame.index))
    if dtype is not None:
        value = value.fillna(default).astype(dtype)
    return value


def _diameter(frame: pd.DataFrame, fallback: float) -> pd.Series:
    diameter = _series(frame, "diameter_px", fallback, float)
    if "area_px" in frame:
        area_diameter = 2.0 * np.sqrt(
            np.maximum(_series(frame, "area_px", 0.0, float), 0.0) / np.pi
        )
        diameter = np.maximum(diameter, area_diameter)
    return pd.Series(np.clip(diameter, 3.0, 160.0), index=frame.index)


def _protected(frame: pd.DataFrame) -> pd.Series:
    reviewed = _series(frame, "reviewed_label", "").fillna("").astype(str).ne("")
    return (
        reviewed
        | _series(frame, "is_manual_missed", False, bool)
        | _series(frame, "manual_override", False, bool)
        | _series(frame, "candidate_source", "").astype(str).eq(
            "manual_cell_anchor"
        )
    )


def _source_priority(source: str) -> int:
    return {
        "manual_cell_anchor": 6,
        "instrument_csv": 5,
        "cf_component": 4,
        "wall_residual_peak": 3,
        "wall_cell_rescue_peak": 3,
        "multiscale_dense_peak": 2,
    }.get(source, 1)


def _assign_component_footprints(
    result: pd.DataFrame,
    settings: dict[str, Any],
    diameters: pd.Series,
) -> pd.DataFrame:
    """Attach raw/residual peaks to the compact CF component they sample.

    A dense detector reports a local maximum, not an object boundary.  The CF
    component is the currently available full-object mask.  Associating every
    nearby peak with that component lets later consolidation operate on an
    instance footprint rather than on a fragile centre-distance threshold.
    """
    output = result.copy()
    output["instance_component_id"] = ""
    output["instance_component_distance_px"] = np.nan
    output["instance_footprint_diameter_px"] = diameters.astype(float)
    if "candidate_source" not in output:
        return output

    base_radius = float(settings.get("component_assignment_minimum_px", 6.0))
    diameter_scale = float(settings.get("component_assignment_diameter_scale", 0.75))
    margin = float(settings.get("component_assignment_margin_px", 3.0))
    maximum_radius = float(settings.get("component_assignment_maximum_px", 16.0))
    maximum_component_area = float(settings.get("component_maximum_area_px", 1200.0))
    maximum_component_diameter = float(settings.get("component_maximum_diameter_px", 48.0))

    sources = output["candidate_source"].fillna("").astype(str)
    areas = _series(output, "area_px", 0.0, float)
    for _, local_indices in output.groupby(["well", "timepoint"], sort=False).groups.items():
        local_indices = list(local_indices)
        anchors = [
            index
            for index in local_indices
            if sources.at[index] == "cf_component"
            and 3.0 <= areas.at[index] <= maximum_component_area
            and diameters.at[index] <= maximum_component_diameter
        ]
        if not anchors:
            continue
        anchor_xy = output.loc[anchors, ["x_px", "y_px"]].to_numpy(float)
        for anchor in anchors:
            output.at[anchor, "instance_component_id"] = str(
                output.at[anchor, "candidate_id"]
            )
            output.at[anchor, "instance_component_distance_px"] = 0.0
        for index in local_indices:
            if index in anchors:
                continue
            point = output.loc[index, ["x_px", "y_px"]].to_numpy(float)
            distances = np.linalg.norm(anchor_xy - point, axis=1)
            position = int(np.argmin(distances))
            anchor = anchors[position]
            distance = float(distances[position])
            radius = min(
                maximum_radius,
                max(
                    base_radius,
                    diameter_scale * float(diameters.at[anchor]) + margin,
                    0.35 * float(diameters.at[index])
                    + 0.50 * float(diameters.at[anchor]),
                ),
            )
            if distance > radius:
                continue
            output.at[index, "instance_component_id"] = str(
                output.at[anchor, "candidate_id"]
            )
            output.at[index, "instance_component_distance_px"] = distance
            output.at[index, "instance_footprint_diameter_px"] = float(
                diameters.at[anchor]
            )
    return output


def _same_instance(
    left: int,
    right: int,
    result: pd.DataFrame,
    labels: pd.Series,
    diameters: pd.Series,
    settings: dict[str, Any],
) -> bool:
    left_label = labels.at[left]
    right_label = labels.at[right]
    if "instance_component_id" in result:
        left_component = str(result.at[left, "instance_component_id"] or "")
        right_component = str(result.at[right, "instance_component_id"] or "")
        if left_component and left_component == right_component:
            return True
    if left_label != right_label:
        # Doublet and 3+ predictions are alternative representations of one
        # group, while a single/group pair is handled as a parent-child case.
        if not ({left_label, right_label} <= GROUP_LABELS):
            return False
    if left_label not in CELL_LABELS | {"debris"}:
        return False

    dx = float(result.at[left, "x_px"]) - float(result.at[right, "x_px"])
    dy = float(result.at[left, "y_px"]) - float(result.at[right, "y_px"])
    distance = float(np.hypot(dx, dy))
    d1, d2 = float(diameters.at[left]), float(diameters.at[right])
    source1 = str(result.at[left, "candidate_source"]) if "candidate_source" in result else ""
    source2 = str(result.at[right, "candidate_source"]) if "candidate_source" in result else ""

    base = float(settings.get("duplicate_minimum_radius_px", 4.0))
    if left_label == "single" and right_label == "single":
        # Adjacent true cells can be only one diameter apart.  Singles are
        # merged only when their centres occupy the same central footprint;
        # cross-source agreement permits a slightly wider tolerance.
        fraction = float(settings.get("single_duplicate_diameter_fraction", 0.42))
        radius = max(base, fraction * min(d1, d2))
        if source1 != source2:
            radius += float(settings.get("cross_source_radius_bonus_px", 2.0))
        radius = min(radius, float(settings.get("single_duplicate_maximum_px", 9.0)))
    elif left_label in GROUP_LABELS and right_label in GROUP_LABELS:
        fraction = float(settings.get("group_duplicate_diameter_fraction", 0.60))
        radius = max(base + 2.0, fraction * max(d1, d2))
        radius = min(radius, float(settings.get("group_duplicate_maximum_px", 28.0)))
    else:
        fraction = float(settings.get("debris_duplicate_diameter_fraction", 0.45))
        radius = min(
            max(base, fraction * min(d1, d2)),
            float(settings.get("debris_duplicate_maximum_px", 12.0)),
        )
    return distance <= radius


def _choose_representative(
    indices: list[int],
    result: pd.DataFrame,
    confidence: pd.Series,
    protected: pd.Series,
    labels: pd.Series,
) -> int:
    def score(index: int) -> tuple[float, float, float, float]:
        label = labels.at[index]
        label_specific = 0.0
        probability_column = {
            "single": "single_probability",
            "touching_doublet": "touching_doublet_probability",
            "cluster_3plus": "cluster_3plus_probability",
            "debris": "debris_probability",
        }.get(label)
        if probability_column and probability_column in result:
            label_specific = float(result.at[index, probability_column] or 0.0)
        source = (
            str(result.at[index, "candidate_source"])
            if "candidate_source" in result
            else ""
        )
        return (
            float(protected.at[index]),
            float(_source_priority(source)),
            max(float(confidence.at[index]), label_specific),
            float(result.at[index, "dense_response"] or 0.0)
            if "dense_response" in result
            else 0.0,
        )

    return max(indices, key=score)


def suppress_nested_single_candidates(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    label_column: str,
    confidence_column: str,
) -> pd.DataFrame:
    """Consolidate proposals into mutually exclusive biological instances.

    The function performs two deliberately separate operations: duplicate
    proposals of the same object are collapsed first; then single-cell cores
    contained by a confident doublet/cluster are attached to that parent.
    Original labels stay available for audit and no row is deleted.
    """
    result = frame.copy()
    result["is_duplicate_suppressed"] = False
    result["duplicate_of_candidate_id"] = ""
    result["duplicate_suppression_reason"] = ""
    result["is_hierarchy_suppressed"] = False
    result["suppressed_by_candidate_id"] = ""
    result["hierarchy_suppression_reason"] = ""
    result["parent_candidate_id"] = ""
    result["is_counting_instance"] = True
    if result.empty or label_column not in result:
        return result

    settings = config.get("hierarchical_suppression", {})
    minimum_group_confidence = float(settings.get("minimum_group_confidence", 0.58))
    fallback_single_diameter = float(settings.get("single_diameter_reference_px", 8.0))
    doublet_scale = float(settings.get("doublet_radius_scale", 2.0))
    cluster_scale = float(settings.get("cluster_radius_scale", 2.8))
    maximum_radius = float(settings.get("maximum_radius_px", 96.0))

    labels = result[label_column].astype(str)
    confidence = _series(result, confidence_column, 0.0, float)
    protected = _protected(result)
    diameters = _diameter(result, fallback_single_diameter)
    result = _assign_component_footprints(result, settings, diameters)
    single_values = diameters[labels.eq("single")]
    single_reference = (
        float(np.clip(single_values.median(), 6.0, 18.0))
        if len(single_values)
        else fallback_single_diameter
    )

    for _, grouped_indices in result.groupby(["well", "timepoint"], sort=False).groups.items():
        local_indices = list(grouped_indices)

        # Connected components of near-identical proposals.  A protected human
        # point is never silently absorbed by another protected point.
        remaining = {
            index for index in local_indices if labels.at[index] in CELL_LABELS | {"debris"}
        }
        components: list[list[int]] = []
        while remaining:
            seed = remaining.pop()
            component = [seed]
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbours = [
                    other
                    for other in list(remaining)
                    if not (bool(protected.at[current]) and bool(protected.at[other]))
                    and _same_instance(current, other, result, labels, diameters, settings)
                ]
                for other in neighbours:
                    remaining.remove(other)
                    component.append(other)
                    frontier.append(other)
            components.append(component)

        for component in components:
            if len(component) < 2:
                continue
            chosen = _choose_representative(component, result, confidence, protected, labels)
            for duplicate in component:
                if duplicate == chosen:
                    continue
                result.at[duplicate, "is_duplicate_suppressed"] = True
                result.at[duplicate, "duplicate_of_candidate_id"] = str(
                    result.at[chosen, "candidate_id"]
                )
                result.at[duplicate, "duplicate_suppression_reason"] = (
                    f"adaptive_same_instance_{labels.at[duplicate]}"
                )
                result.at[duplicate, "is_counting_instance"] = False

        active = [
            index for index in local_indices if not bool(result.at[index, "is_duplicate_suppressed"])
        ]
        anchors = [
            index
            for index in active
            if labels.at[index] in GROUP_LABELS
            and (confidence.at[index] >= minimum_group_confidence or bool(protected.at[index]))
        ]
        singles = [index for index in active if labels.at[index] == "single"]
        if not anchors or not singles:
            continue

        for single_index in singles:
            sx = float(result.at[single_index, "x_px"])
            sy = float(result.at[single_index, "y_px"])
            matches: list[tuple[float, int, float]] = []
            for anchor_index in anchors:
                group_label = labels.at[anchor_index]
                group_diameter = float(diameters.at[anchor_index])
                class_radius = single_reference * (
                    doublet_scale if group_label == "touching_doublet" else cluster_scale
                )
                # The parent contour and the child's centre must overlap.  The
                # small allowance covers segmentation centring error without
                # swallowing a genuinely adjacent independent cell.
                footprint_radius = group_diameter / 2.0 + 0.55 * single_reference
                radius = min(maximum_radius, max(class_radius, footprint_radius))
                distance = float(
                    np.hypot(
                        sx - float(result.at[anchor_index, "x_px"]),
                        sy - float(result.at[anchor_index, "y_px"]),
                    )
                )
                if distance <= radius:
                    matches.append((distance / max(radius, 1.0), anchor_index, radius))
            if not matches:
                continue
            _, chosen, radius = min(matches, key=lambda item: item[0])
            result.at[single_index, "is_hierarchy_suppressed"] = True
            result.at[single_index, "suppressed_by_candidate_id"] = str(
                result.at[chosen, "candidate_id"]
            )
            result.at[single_index, "parent_candidate_id"] = str(
                result.at[chosen, "candidate_id"]
            )
            result.at[single_index, "hierarchy_suppression_reason"] = (
                f"single_core_inside_{labels.at[chosen]}_footprint_r{radius:.1f}"
            )
            result.at[single_index, "is_counting_instance"] = False

    return result
