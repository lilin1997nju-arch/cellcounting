from __future__ import annotations

import math
from typing import Any


CELL_LABELS = {"single", "touching_doublet", "cluster_3plus"}
MULTIPLICITY_PROBABILITIES = {
    "single": "single_probability",
    "touching_doublet": "touching_doublet_probability",
    "cluster_3plus": "cluster_3plus_probability",
}


def _text(value: Any) -> str:
    result = str(value or "")
    return "" if result.lower() in {"nan", "none"} else result


def candidate_multiplicity_label(candidate: Any) -> str:
    """Return the candidate's own best biological multiplicity.

    Temporal evidence is deliberately excluded: it may decide whether the
    object is a cell, but it must not copy another frame's multiplicity onto
    this candidate.
    """

    reviewed = _text(candidate.get("reviewed_label", ""))
    if reviewed in CELL_LABELS:
        return reviewed

    integrated = _text(candidate.get("integrated_label", ""))
    for column in (
        "v2_pre_temporal_integrated_label",
        "v2_original_integrated_label",
    ):
        label = _text(candidate.get(column, ""))
        if label in CELL_LABELS:
            return label

    # ``integrated_label`` may already contain a temporal write-back, even on
    # rows carrying the historical manual_override flag.  Only use the current
    # value after the immutable pre-temporal labels have been exhausted.
    if bool(candidate.get("manual_override", False)) and integrated in CELL_LABELS:
        return integrated

    predicted = _text(candidate.get("predicted_multiplicity", ""))
    if predicted in CELL_LABELS:
        return predicted

    probabilities: dict[str, float] = {}
    for label, column in MULTIPLICITY_PROBABILITIES.items():
        try:
            value = float(candidate.get(column, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            probabilities[label] = value
    if probabilities:
        return max(probabilities, key=probabilities.get)

    if integrated in CELL_LABELS:
        return integrated
    return "single"
