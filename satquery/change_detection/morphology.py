"""
satquery.change_detection.morphology
======================================
Post-processing morphological operations to clean binary change masks.

Operations applied in sequence
--------------------------------
1. **Morphological opening** (erosion → dilation)
   Removes isolated single-pixel and small-cluster noise hits from the
   binary change mask.  Disk radius controls the minimum connected area
   retained.

2. **Morphological closing** (dilation → erosion)
   Fills small holes inside change polygons, producing more solid,
   connected change regions that are easier to polygonize.

3. **Small-component removal**
   Finds all 8-connected components and removes any with fewer than
   *min_area_px* pixels. This eliminates residual salt-and-pepper noise
   that Otsu did not fully separate.

All operations use OpenCV for ABI stability (no scipy/skimage ABI required).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _disk_kernel(radius: int) -> np.ndarray:
    """Create a circular structuring element of given radius (OpenCV-compatible)."""
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )


def clean_mask(
    mask: np.ndarray,
    open_radius: int = 2,
    close_radius: int = 3,
    min_area_px: int = 25,
) -> np.ndarray:
    """
    Apply the full morphological cleaning pipeline to *mask*.

    Parameters
    ----------
    mask          : (H, W) bool / uint8 binary change mask.
    open_radius   : Disk radius for morphological opening (noise removal).
                    Set to 0 to disable opening.
    close_radius  : Disk radius for morphological closing (hole-filling).
                    Set to 0 to disable closing.
    min_area_px   : Connected components smaller than this are discarded.

    Returns
    -------
    Cleaned (H, W) bool mask.
    """
    cleaned = mask.astype(np.uint8)

    # --- Opening: remove small speckle noise ---
    if open_radius > 0:
        before = int(cleaned.sum())
        kernel  = _disk_kernel(open_radius)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        after   = int(cleaned.sum())
        logger.debug("Morphological opening (r=%d): %d → %d changed px", open_radius, before, after)

    # --- Closing: fill small interior holes ---
    if close_radius > 0:
        before  = int(cleaned.sum())
        kernel  = _disk_kernel(close_radius)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        after   = int(cleaned.sum())
        logger.debug("Morphological closing (r=%d): %d → %d changed px", close_radius, before, after)

    # --- Small-component removal via connectedComponentsWithStats ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8, ltype=cv2.CV_32S
    )
    before = int(cleaned.sum())
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area_px:
            cleaned[labels == lbl] = 0

    after = int(cleaned.sum())
    n_removed = before - after
    if n_removed > 0:
        logger.debug(
            "Small-component removal (min=%d px): %d px removed.", min_area_px, n_removed
        )

    return cleaned.astype(bool)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Label connected components in *mask* using 8-connectivity (OpenCV).

    Returns
    -------
    (labeled_array, n_components)
        labeled_array : (H, W) int32 array; 0 = background, 1…N = components.
        n_components  : Total number of distinct change regions.
    """
    num_labels, labeled = cv2.connectedComponents(
        mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    n = num_labels - 1  # subtract background label
    logger.debug("Connected-component labelling: %d regions found.", n)
    return labeled, max(n, 0)


def get_component_stats(labeled: np.ndarray) -> list[dict]:
    """
    Compute per-component statistics from a labelled mask.

    Returns
    -------
    List of dicts, one per component, sorted descending by area:
        {"label": int, "area_px": int, "centroid_row": float, "centroid_col": float}
    """
    stats: list[dict] = []
    n = int(labeled.max())
    for lbl in range(1, n + 1):
        comp = labeled == lbl
        area = int(comp.sum())
        rows, cols = np.where(comp)
        stats.append(
            {
                "label":        lbl,
                "area_px":      area,
                "centroid_row": float(rows.mean()) if rows.size > 0 else 0.0,
                "centroid_col": float(cols.mean()) if cols.size > 0 else 0.0,
            }
        )
    stats.sort(key=lambda x: x["area_px"], reverse=True)
    return stats
