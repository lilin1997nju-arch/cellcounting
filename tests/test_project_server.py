from __future__ import annotations

import pytest
from fastapi import HTTPException

from cellvision.project_server import _default_timepoint_selection, _validate_timepoint_selection


def _options():
    return [
        {"day_label": "Day0", "day_number": 0, "timepoint_labels": ["T0"], "eligible_endpoint": False},
        {"day_label": "Day1", "day_number": 1, "timepoint_labels": ["T1"], "eligible_endpoint": False},
        {"day_label": "Day2", "day_number": 2, "timepoint_labels": ["T2"], "eligible_endpoint": False},
        {"day_label": "Day7", "day_number": 7, "timepoint_labels": ["T3"], "eligible_endpoint": True},
        {"day_label": "Day14", "day_number": 14, "timepoint_labels": ["T4"], "eligible_endpoint": True},
    ]


def test_default_selection_uses_latest_late_day():
    assert _default_timepoint_selection(_options()) == ["Day0", "Day1", "Day2", "Day14"]
    assert _validate_timepoint_selection(_options(), ["Day0", "Day1", "Day2", "Day7"]) == (
        ["Day0", "Day1", "Day2", "Day7"],
        "Day7",
        7,
    )


@pytest.mark.parametrize("selected", [["Day0", "Day1"], ["Day0", "Day1", "Day2"]])
def test_selection_requires_early_chain_and_late_endpoint(selected):
    with pytest.raises(HTTPException):
        _validate_timepoint_selection(_options(), selected)
