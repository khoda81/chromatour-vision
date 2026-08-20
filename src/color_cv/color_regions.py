"""Dominant color extraction and region proposal helpers.

These functions intentionally keep the user in the loop: extract candidate
colors, then let the UI show masks that the user can accept, merge, or remove.
"""

from __future__ import annotations

import numpy as np


def quantize_oklab_histogram(oklab: np.ndarray, bins: tuple[int, int, int] = (32, 32, 32)):
    """Return dominant OKLab histogram bins.

    The histogram is computed in a bounded normalized space. This is not meant
    as the final clustering algorithm; it is a predictable palette proposal
    mechanism for interactive correction.
    """
    pixels = oklab.reshape(-1, 3)
    pixels = pixels[np.isfinite(pixels).all(axis=1)]

    if len(pixels) == 0:
        return []

    # Reasonable OKLab ranges for display colors.
    ranges = ((0.0, 1.0), (-0.5, 0.5), (-0.5, 0.5))
    scaled = []
    for channel, (low, high) in zip(pixels.T, ranges):
        scaled.append(np.clip((channel - low) / (high - low), 0, 1))

    hist, edges = np.histogramdd(np.stack(scaled, axis=1), bins=bins)
    flat = np.argsort(hist.ravel())[::-1]

    result = []
    for index in flat:
        count = hist.ravel()[index]
        if count == 0:
            break

        coords = np.unravel_index(index, hist.shape)
        center = []
        for c, e in zip(coords, edges):
            center.append((e[c] + e[c + 1]) / 2)

        center = np.array(center)
        center[0] = center[0]
        center[1:] = center[1:] * 1.0 - 0.5
        result.append({"oklab": center, "count": int(count)})

    return result


def color_distance_mask(oklab: np.ndarray, color: np.ndarray, threshold: float):
    """Highlight pixels close to a target OKLab color."""
    distance = np.linalg.norm(oklab - color, axis=-1)
    return distance <= threshold, distance


def connected_regions(mask: np.ndarray):
    """Return connected components using scipy when available."""
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("Install scipy for connected region extraction") from exc

    labels, count = ndimage.label(mask)
    return [(labels == i) for i in range(1, count + 1)]


def region_shape_score(mask: np.ndarray):
    """Estimate how compact/rectangular a region is.

    Useful for rejecting carpet texture and keeping object-like regions.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0

    area = len(xs)
    box_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    return area / box_area
