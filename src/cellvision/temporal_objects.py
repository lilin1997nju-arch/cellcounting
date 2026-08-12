from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.transform import resize

from .v2_instance_dataset import _crop
from .v2_instance_inference import decode_rle


@dataclass(frozen=True)
class TemporalObjectDescriptor:
    """Object-centred appearance that deliberately excludes distant background."""

    index: int
    timepoint: str
    aligned_x: float
    aligned_y: float
    area: float
    equivalent_diameter: float
    raw: np.ndarray
    gradient: np.ndarray
    soft_mask: np.ndarray
    binary_mask: np.ndarray
    quality: float


@dataclass(frozen=True)
class TemporalPairEvidence:
    left: int
    right: int
    identity: float
    static: float
    intensity: float
    gradient: float
    tolerant_shape: float
    area_similarity: float
    distance_px: float
    foreground_quality: float
    kind: str = "continuation"


def _normalise_object(raw: np.ndarray, support: np.ndarray) -> np.ndarray:
    values = raw[support]
    if values.size < 12:
        values = raw.ravel()
    lo, hi = np.percentile(values.astype(np.float32), [2, 98])
    return np.clip((raw.astype(np.float32) - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)


def _soft_mask(binary: np.ndarray) -> np.ndarray:
    if not binary.any():
        return np.zeros_like(binary, dtype=np.float32)
    inside = distance_transform_edt(binary)
    outside = distance_transform_edt(~binary)
    signed = inside - outside
    return (1.0 / (1.0 + np.exp(-signed / 1.35))).astype(np.float32)


def build_object_descriptor(
    image: np.ndarray,
    row: pd.Series,
    *,
    instance_patch_size: int = 96,
    descriptor_size: int = 48,
    image_fill_value: float | None = None,
) -> TemporalObjectDescriptor | None:
    """Build an adaptive object crop; the instance mask is localisation, not truth."""

    try:
        mask = decode_rle(str(row.get("v2_mask_rle", "[]")), instance_patch_size)
    except (ValueError, TypeError):
        return None
    if not mask.any():
        return None
    regions = regionprops(mask.astype(np.uint8))
    if not regions:
        return None
    region = max(regions, key=lambda value: value.area)
    fill_value = int(np.median(image)) if image_fill_value is None else int(image_fill_value)
    raw_patch = _crop(
        image,
        float(row["x_px"]),
        float(row["y_px"]),
        instance_patch_size,
        fill_value,
    )
    center_y, center_x = region.centroid
    equivalent_diameter = max(float(region.equivalent_diameter_area), 3.0)
    crop_size = int(np.clip(np.ceil(equivalent_diameter * 2.6), 24, 64))
    if crop_size % 2:
        crop_size += 1
    local_raw = _crop(raw_patch, center_x, center_y, crop_size, fill_value)
    local_mask = _crop(mask, center_x, center_y, crop_size, False).astype(bool)
    envelope = binary_dilation(local_mask, iterations=max(2, int(round(equivalent_diameter * 0.20))))
    normalised = _normalise_object(local_raw, envelope)

    raw_resized = resize(
        normalised,
        (descriptor_size, descriptor_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    mask_resized = resize(
        local_mask.astype(np.float32),
        (descriptor_size, descriptor_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    binary_resized = mask_resized >= 0.45
    soft = _soft_mask(binary_resized)
    gy, gx = np.gradient(gaussian_filter(raw_resized, 0.65))
    gradient = np.hypot(gx, gy).astype(np.float32)
    gradient_scale = float(np.percentile(gradient[soft >= 0.10], 95)) if np.any(soft >= 0.10) else 0.0
    if gradient_scale > 1e-6:
        gradient = np.clip(gradient / gradient_scale, 0.0, 1.0)

    confidence = float(row.get("v2_instance_confidence", 0.0) or 0.0)
    objectness = float(row.get("v2_objectness", confidence) or 0.0)
    area_quality = float(np.clip(float(region.area) / 24.0, 0.0, 1.0))
    quality = float(np.clip((0.45 * confidence + 0.35 * objectness + 0.20 * area_quality), 0.0, 1.0))
    x_column = "aligned_x_px" if "aligned_x_px" in row.index else "x_px"
    y_column = "aligned_y_px" if "aligned_y_px" in row.index else "y_px"
    return TemporalObjectDescriptor(
        index=int(row.name),
        timepoint=str(row["timepoint"]),
        aligned_x=float(row[x_column]),
        aligned_y=float(row[y_column]),
        area=float(region.area),
        equivalent_diameter=equivalent_diameter,
        raw=raw_resized,
        gradient=gradient,
        soft_mask=soft,
        binary_mask=binary_resized,
        quality=quality,
    )


def _shifted_views(
    first: np.ndarray,
    second: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = first.shape
    first_y0, first_y1 = max(0, -dy), min(height, height - dy)
    first_x0, first_x1 = max(0, -dx), min(width, width - dx)
    second_y0, second_y1 = first_y0 + dy, first_y1 + dy
    second_x0, second_x1 = first_x0 + dx, first_x1 + dx
    return (
        first[first_y0:first_y1, first_x0:first_x1],
        second[second_y0:second_y1, second_x0:second_x1],
    )


def _weighted_correlation(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> float:
    active = weights > 0.05
    if int(active.sum()) < 12:
        return 0.0
    values_a = first[active].astype(np.float64)
    values_b = second[active].astype(np.float64)
    local_weights = weights[active].astype(np.float64)
    local_weights /= max(float(local_weights.sum()), 1e-8)
    values_a -= float(np.sum(values_a * local_weights))
    values_b -= float(np.sum(values_b * local_weights))
    denominator = np.sqrt(
        float(np.sum(values_a * values_a * local_weights))
        * float(np.sum(values_b * values_b * local_weights))
    )
    if denominator <= 1e-8:
        return 0.0
    return float(np.clip(np.sum(values_a * values_b * local_weights) / denominator, 0.0, 1.0))


def compare_object_descriptors(
    first: TemporalObjectDescriptor,
    second: TemporalObjectDescriptor,
    *,
    maximum_shift: int = 3,
    correspondence_distance_px: float = 128.0,
) -> TemporalPairEvidence:
    """Compare actual foreground; unchanged surrounding texture contributes zero."""

    best: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    for dy in range(-maximum_shift, maximum_shift + 1):
        for dx in range(-maximum_shift, maximum_shift + 1):
            raw_a, raw_b = _shifted_views(first.raw, second.raw, dy, dx)
            grad_a, grad_b = _shifted_views(first.gradient, second.gradient, dy, dx)
            soft_a, soft_b = _shifted_views(first.soft_mask, second.soft_mask, dy, dx)
            mask_a, mask_b = _shifted_views(first.binary_mask, second.binary_mask, dy, dx)
            # Most weight is shared foreground. A small union term tolerates an
            # under/over-segmented boundary without letting distant background in.
            weights = np.clip(np.sqrt(soft_a * soft_b) + 0.12 * np.maximum(soft_a, soft_b), 0.0, 1.0)
            intensity = _weighted_correlation(raw_a, raw_b, weights)
            gradient = _weighted_correlation(grad_a, grad_b, weights)
            dilated_a = binary_dilation(mask_a, iterations=2)
            dilated_b = binary_dilation(mask_b, iterations=2)
            coverage_a = float(np.logical_and(mask_a, dilated_b).sum()) / max(float(mask_a.sum()), 1.0)
            coverage_b = float(np.logical_and(mask_b, dilated_a).sum()) / max(float(mask_b.sum()), 1.0)
            tolerant_shape = float(np.clip((coverage_a + coverage_b) / 2.0, 0.0, 1.0))
            combined = 0.56 * intensity + 0.24 * gradient + 0.20 * tolerant_shape
            if combined > best[0]:
                best = (combined, intensity, gradient, tolerant_shape)

    appearance, intensity, gradient, tolerant_shape = best
    area_similarity = float(
        np.exp(-abs(np.log((second.area + 1.0) / (first.area + 1.0))) / 0.85)
    )
    distance = float(
        np.hypot(second.aligned_x - first.aligned_x, second.aligned_y - first.aligned_y)
    )
    distance_similarity = float(np.exp(-((distance / max(correspondence_distance_px * 0.72, 1.0)) ** 2)))
    quality = min(first.quality, second.quality)
    identity = float(
        np.clip(
            0.52 * appearance
            + 0.20 * area_similarity
            + 0.23 * distance_similarity
            + 0.05 * quality,
            0.0,
            1.0,
        )
    )
    static = float(
        np.clip(
            # Exact boundaries and their gradients vary most when the instance
            # mask is slightly over/under-segmented. Foreground intensity and
            # tolerant shape therefore dominate the static-object decision.
            0.65 * intensity
            + 0.08 * gradient
            + 0.20 * tolerant_shape
            + 0.07 * area_similarity,
            0.0,
            1.0,
        )
    )
    return TemporalPairEvidence(
        left=first.index,
        right=second.index,
        identity=identity,
        static=static,
        intensity=intensity,
        gradient=gradient,
        tolerant_shape=tolerant_shape,
        area_similarity=area_similarity,
        distance_px=distance,
        foreground_quality=quality,
    )


def match_timepoint_objects(
    left: list[TemporalObjectDescriptor],
    right: list[TemporalObjectDescriptor],
    *,
    maximum_distance_px: float,
    minimum_identity: float,
    maximum_shift: int = 3,
    pairwise_scorer: Callable[[TemporalObjectDescriptor, TemporalObjectDescriptor, TemporalPairEvidence], TemporalPairEvidence] | None = None,
) -> tuple[list[TemporalPairEvidence], dict[tuple[int, int], TemporalPairEvidence]]:
    """Globally match objects one-to-one while allowing every object to remain unmatched."""

    if not left or not right:
        return [], {}
    scores = np.full((len(left), len(right)), -1.0, dtype=np.float64)
    evidence: dict[tuple[int, int], TemporalPairEvidence] = {}
    for left_index, first in enumerate(left):
        for right_index, second in enumerate(right):
            distance = float(np.hypot(second.aligned_x - first.aligned_x, second.aligned_y - first.aligned_y))
            if distance > maximum_distance_px:
                continue
            pair = compare_object_descriptors(
                first,
                second,
                maximum_shift=maximum_shift,
                correspondence_distance_px=maximum_distance_px,
            )
            if pairwise_scorer is not None:
                pair = pairwise_scorer(first, second, pair)
            scores[left_index, right_index] = pair.identity
            evidence[(first.index, second.index)] = pair
    if np.all(scores < 0):
        return [], evidence
    row_indices, column_indices = linear_sum_assignment(np.where(scores >= 0, 1.0 - scores, 2.0))
    matches = [
        evidence[(left[row].index, right[column].index)]
        for row, column in zip(row_indices, column_indices)
        if scores[row, column] >= minimum_identity
    ]
    return matches, evidence


class TemporalComponents:
    def __init__(self, nodes: list[int]):
        self.parent = {int(node): int(node) for node in nodes}

    def find(self, node: int) -> int:
        root = int(node)
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[int(node)] != int(node):
            previous = self.parent[int(node)]
            self.parent[int(node)] = root
            node = previous
        return root

    def union(self, first: int, second: int) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def groups(self) -> list[list[int]]:
        output: dict[int, list[int]] = {}
        for node in self.parent:
            output.setdefault(self.find(node), []).append(node)
        return list(output.values())


def multiplicity_rank(label: str) -> int:
    return {
        "single": 1,
        "touching_doublet": 2,
        "cluster_3plus": 3,
    }.get(str(label), 1)


def conditional_cell_probability(row: pd.Series) -> float:
    cell = float(row.get("cell_probability", 0.0) or 0.0)
    debris = float(row.get("debris_probability", 0.0) or 0.0)
    return float(cell / max(cell + debris, 1e-8))


def detect_division_edges(
    frame: pd.DataFrame,
    left: list[TemporalObjectDescriptor],
    right: list[TemporalObjectDescriptor],
    matches: list[TemporalPairEvidence],
    all_evidence: dict[tuple[int, int], TemporalPairEvidence],
    *,
    division_radius_px: float = 80.0,
) -> list[TemporalPairEvidence]:
    """Allow a conservative parent-to-many hypothesis for proliferation evidence."""

    matched_right = {pair.right for pair in matches}
    match_by_left = {pair.left: pair for pair in matches}
    output: list[TemporalPairEvidence] = []
    right_lookup = {item.index: item for item in right}
    for parent in left:
        parent_row = frame.loc[parent.index]
        if conditional_cell_probability(parent_row) < 0.22:
            continue
        children: list[TemporalObjectDescriptor] = []
        existing = match_by_left.get(parent.index)
        if existing is not None:
            children.append(right_lookup[existing.right])
        available = [
            child
            for child in right
            if child.index not in matched_right
            and float(np.hypot(child.aligned_x - parent.aligned_x, child.aligned_y - parent.aligned_y))
            <= division_radius_px
            and conditional_cell_probability(frame.loc[child.index]) >= 0.22
        ]
        available.sort(
            key=lambda child: float(np.hypot(child.aligned_x - parent.aligned_x, child.aligned_y - parent.aligned_y))
        )
        children.extend(available[: max(0, 3 - len(children))])
        if len(children) < 2:
            continue
        combined_area_ratio = sum(child.area for child in children) / max(parent.area, 1.0)
        centroid_x = np.average([child.aligned_x for child in children], weights=[child.area for child in children])
        centroid_y = np.average([child.aligned_y for child in children], weights=[child.area for child in children])
        centroid_distance = float(np.hypot(centroid_x - parent.aligned_x, centroid_y - parent.aligned_y))
        if not (0.65 <= combined_area_ratio <= 5.0 and centroid_distance <= division_radius_px):
            continue
        pair_values = []
        for child in children:
            pair = all_evidence.get((parent.index, child.index))
            if pair is None:
                pair = compare_object_descriptors(
                    parent,
                    child,
                    correspondence_distance_px=division_radius_px,
                )
            pair_values.append(pair)
        if max(pair.identity for pair in pair_values) < 0.48:
            continue
        for pair in pair_values:
            output.append(
                TemporalPairEvidence(
                    **{**pair.__dict__, "identity": max(pair.identity, 0.58), "kind": "division"}
                )
            )
            matched_right.add(pair.right)
    return output


def component_path_confidence(
    source: int,
    target: int,
    edges: list[TemporalPairEvidence],
) -> float:
    if source == target:
        return 1.0
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left, []).append((edge.right, edge.identity))
        adjacency.setdefault(edge.right, []).append((edge.left, edge.identity))
    best = {source: 1.0}
    pending = [source]
    while pending:
        current = pending.pop()
        for neighbour, confidence in adjacency.get(current, []):
            candidate = best[current] * float(confidence)
            if candidate > best.get(neighbour, 0.0):
                best[neighbour] = candidate
                pending.append(neighbour)
    return float(best.get(target, 0.0))
