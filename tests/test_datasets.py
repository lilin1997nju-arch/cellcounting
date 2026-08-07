import numpy as np

from cellvision.datasets import synchronous_transform


def test_timepoints_receive_same_spatial_transform():
    base = np.arange(16).reshape(4, 4)
    images, masks = synchronous_transform([base, base + 100], [base > 5, base > 5], 1, True)
    assert np.array_equal(images[1] - images[0], np.full((4, 4), 100))
    assert np.array_equal(masks[0], masks[1])

