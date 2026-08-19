from __future__ import annotations

import json
import os

from cellvision.review_summary import (
    SUMMARY_VERSION,
    latest_prediction_path,
    read_cached_summary,
    read_summary,
    summary_path,
    summary_signature,
    write_summary,
)


def test_latest_prediction_path_prefers_newest_supported_result(tmp_path):
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    old = prediction_root / "latest_v2_predictions.csv"
    new = prediction_root / "latest_v3_predictions.csv"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    old_ns = old.stat().st_mtime_ns
    os.utime(new, ns=(old_ns + 10_000_000_000, old_ns + 10_000_000_000))

    assert latest_prediction_path(tmp_path) == new


def test_summary_round_trip_requires_matching_signature(tmp_path):
    (tmp_path / "predictions").mkdir()
    prediction = tmp_path / "predictions" / "latest_v3_predictions.csv"
    prediction.write_text("candidate_id\n1\n", encoding="utf-8")
    signature = summary_signature(tmp_path)
    path = summary_path(tmp_path)
    payload = {
        "version": SUMMARY_VERSION,
        "signature": signature,
        "status": "ready",
        "wells": [],
    }

    write_summary(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert read_summary(path, signature) == payload
    changed = dict(signature, prediction_mtime_ns=(signature["prediction_mtime_ns"] or 0) + 1)
    assert read_summary(path, changed) is None
    assert read_cached_summary(path) == payload
