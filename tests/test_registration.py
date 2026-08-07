import numpy as np

from cellvision.registration import phase_correlation_shift, pixels_to_micrometers


def test_pixel_to_micrometers():
    assert pixels_to_micrometers(10, 2.08) == 20.8


def test_phase_correlation_translation():
    reference = np.zeros((64, 64), dtype=np.float32)
    reference[20:25, 30:36] = 1
    moving = np.roll(np.roll(reference, 4, axis=0), -3, axis=1)
    x_shift, y_shift = phase_correlation_shift(reference, moving)
    assert (x_shift, y_shift) == (3.0, -4.0)

