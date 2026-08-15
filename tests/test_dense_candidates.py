import numpy as np

import cellvision.dense_candidates as dense_candidates
from cellvision.dense_candidates import (
    _polar_wall_residual_peaks,
    _select_response_backend,
    _zone_peak_indices,
)


def test_multiscale_backend_auto_uses_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        dense_candidates,
        "_CUDA_STATUS",
        (True, "test-gpu"),
    )

    assert _select_response_backend({"multiscale_backend": "auto"}) == (
        "cuda",
        "test-gpu",
    )


def test_multiscale_backend_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        dense_candidates,
        "_CUDA_STATUS",
        (False, "cuda_unavailable"),
    )

    assert _select_response_backend({"multiscale_backend": "auto"}) == (
        "cpu",
        "cuda_unavailable",
    )


def test_zone_peak_indices_do_not_fill_empty_texture_tiles() -> None:
    response = np.zeros((32, 32), dtype=np.float32)
    response[4, 4] = 7.0
    response[12, 12] = 9.0
    response[20, 20] = 11.0
    maxima = response > 0
    zone = np.ones_like(maxima, dtype=bool)

    selected = _zone_peak_indices(
        response,
        maxima,
        zone,
        percentile=90.0,
        minimum_response=18.0,
        quota=16,
        grid_divisions=4,
        coverage_per_tile=2,
    )

    assert selected == []


def test_zone_peak_indices_keep_salient_peaks_without_forcing_weak_ones() -> None:
    response = np.zeros((32, 32), dtype=np.float32)
    response[4, 4] = 8.0
    response[12, 12] = 24.0
    response[20, 20] = 42.0
    maxima = response > 0
    zone = np.ones_like(maxima, dtype=bool)

    selected = _zone_peak_indices(
        response,
        maxima,
        zone,
        percentile=80.0,
        minimum_response=18.0,
        quota=16,
        grid_divisions=4,
        coverage_per_tile=2,
    )

    assert set(selected) == {(12, 12), (20, 20)}


def test_polar_wall_residual_removes_rim_but_keeps_overlapping_blob() -> None:
    size = 256
    yy, xx = np.indices((size, size))
    radius = np.hypot(xx - size / 2, yy - size / 2)
    wall = np.full((size, size), 110.0, dtype=np.float32)
    wall[np.abs(radius - 108.0) <= 2.0] = 45.0
    settings = {
        "wall_residual_enabled": True,
        "wall_residual_inside_fraction": 0.05,
        "wall_residual_outside_fraction": 0.05,
        "wall_residual_maximum_fraction": 0.48,
        "wall_residual_angular_samples": 1024,
        "wall_residual_tangential_sigma": 10.0,
        "wall_residual_minimum_response": 18.0,
        "wall_residual_candidate_quota": 12,
        "wall_residual_radial_window": 9,
        "wall_residual_angular_window": 7,
        "duplicate_radius_px": 5,
    }

    assert _polar_wall_residual_peaks(wall, 108 / size, settings) == []

    overlapping = wall.copy()
    cell = (xx - 236) ** 2 + (yy - 128) ** 2 <= 5**2
    overlapping[cell] = 5.0
    peaks = _polar_wall_residual_peaks(
        overlapping, 108 / size, settings
    )

    assert any(
        np.hypot(peak["x_px"] - 236, peak["y_px"] - 128) <= 6
        for peak in peaks
    )
