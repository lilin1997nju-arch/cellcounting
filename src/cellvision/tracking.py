from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackPoint:
    object_id: str
    x: float
    y: float
    area: float = 1.0


def one_to_one_match(
    parents: list[TrackPoint], children: list[TrackPoint], maximum_distance: float
) -> list[tuple[str, str, float]]:
    if not parents or not children:
        return []
    costs = np.empty((len(parents), len(children)), dtype=float)
    for row, parent in enumerate(parents):
        for column, child in enumerate(children):
            distance = np.hypot(parent.x - child.x, parent.y - child.y)
            area_penalty = abs(np.log(max(child.area, 1e-6) / max(parent.area, 1e-6)))
            costs[row, column] = distance + 0.1 * area_penalty
    parent_indices, child_indices = linear_sum_assignment(costs)
    return [
        (parents[row].object_id, children[column].object_id, float(costs[row, column]))
        for row, column in zip(parent_indices, child_indices)
        if costs[row, column] <= maximum_distance
    ]


def division_candidates(
    parent: TrackPoint,
    children: list[TrackPoint],
    maximum_distance: float,
    area_ratio_range: tuple[float, float] = (0.5, 2.5),
) -> list[str]:
    nearby = [
        child
        for child in children
        if np.hypot(parent.x - child.x, parent.y - child.y) <= maximum_distance
    ]
    if len(nearby) < 2:
        return []
    nearby.sort(key=lambda child: np.hypot(parent.x - child.x, parent.y - child.y))
    selected = nearby[:2]
    ratio = sum(child.area for child in selected) / max(parent.area, 1e-6)
    if area_ratio_range[0] <= ratio <= area_ratio_range[1]:
        return [child.object_id for child in selected]
    return []

