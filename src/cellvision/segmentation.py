from __future__ import annotations

import numpy as np
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed
from scipy import ndimage


def split_touching(mask: np.ndarray, minimum_distance: int = 3) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    distance = ndimage.distance_transform_edt(binary)
    peaks = peak_local_max(distance, labels=binary, min_distance=minimum_distance)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for index, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = index
    markers = label(markers > 0)
    return watershed(-distance, markers, mask=binary)

