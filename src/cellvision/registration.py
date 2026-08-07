from __future__ import annotations

import numpy as np


def phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    if reference.shape != moving.shape:
        raise ValueError("Images must have equal shape")
    reference_fft = np.fft.fft2(reference.astype(np.float32))
    moving_fft = np.fft.fft2(moving.astype(np.float32))
    cross_power = reference_fft * np.conjugate(moving_fft)
    cross_power /= np.maximum(np.abs(cross_power), 1e-8)
    correlation = np.fft.ifft2(cross_power)
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    shifts = np.asarray(peak, dtype=float)
    midpoint = np.asarray(reference.shape) // 2
    shifts[shifts > midpoint] -= np.asarray(reference.shape)[shifts > midpoint]
    return float(shifts[1]), float(shifts[0])


def pixels_to_micrometers(pixels: float, resolution_um_per_pixel: float) -> float:
    return float(pixels) * float(resolution_um_per_pixel)

