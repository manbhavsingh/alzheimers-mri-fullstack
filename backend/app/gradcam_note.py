"""Compact textual summaries for existing Grad-CAM activation maps."""

import numpy as np


def gradcam_note(cam: np.ndarray, thr: float = 0.6) -> str:
    """Summarize a normalized 2D CAM without interpreting anatomy.

    The centroid and directions refer only to the image coordinate system.
    """
    heatmap = np.asarray(cam, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError("cam must be a 2D array")
    if heatmap.size == 0:
        return "No strongly activated region."

    active = heatmap > thr
    count = int(np.count_nonzero(active))
    if count == 0:
        return "No strongly activated region."

    rows, columns = np.nonzero(active)
    weights = heatmap[active]
    weight_sum = float(weights.sum())
    if weight_sum > 0:
        center_y = float(np.average(rows, weights=weights))
        center_x = float(np.average(columns, weights=weights))
    else:
        center_y = float(rows.mean())
        center_x = float(columns.mean())

    height, width = heatmap.shape

    vertical = (
        "upper" if center_y < height / 3
        else "middle" if center_y < (2 * height) / 3
        else "lower"
    )
    horizontal = (
        "left" if center_x < width / 3
        else "center" if center_x < (2 * width) / 3
        else "right"
    )

    spread_percent = count * 100.0 / heatmap.size
    spread = (
        "focused" if spread_percent < 10
        else "moderate" if spread_percent < 25
        else "wide"
    )
    return (
        f"Activation centroid: {vertical}-{horizontal}. "
        f"Spread: {spread} ({spread_percent:.1f}% of pixels above threshold {thr:.2f}). "
        "Directions are relative to the image, not anatomical."
    )
