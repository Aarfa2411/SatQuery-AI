"""
satquery.change_detection.metrics
===================================
Real-world area calculation, percentage change, and directional
change classification.

Change direction labels
------------------------
The pipeline maps spectral index changes to one of five semantic labels:

    vegetation_loss     — NDVI decreased significantly
    vegetation_gain     — NDVI increased significantly
    water_expansion     — NDWI increased significantly
    water_contraction   — NDWI decreased significantly
    urban_growth        — NDBI increased / NDVI decreased in non-water area
    sar_change          — Generic label for SAR-only change (no optical index)
    mixed_change        — Multiple indices contradict each other
    no_change           — Mask is essentially empty

Classification is heuristic / rule-based and explicitly designed to be
interpretable by non-expert users (problem statement requirement).
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Semantic change labels
ChangeLabel = Literal[
    "vegetation_loss",
    "vegetation_gain",
    "water_expansion",
    "water_contraction",
    "urban_growth",
    "sar_change",
    "mixed_change",
    "no_change",
]

# Minimum fraction of changed pixels required to classify direction
_MIN_CHANGED_FRACTION = 0.001   # 0.1 %


# ---------------------------------------------------------------------------
# Area calculation
# ---------------------------------------------------------------------------

def compute_area(
    mask: np.ndarray,
    transform,
) -> dict[str, float]:
    """
    Compute the real-world area of changed pixels using the raster's
    affine transform.

    Parameters
    ----------
    mask      : (H, W) bool array — True = changed pixel.
    transform : rasterio Affine transform (pixel size encoded in |a| and |e|).

    Returns
    -------
    dict with:
        "area_m2"          : float — total changed area in square metres.
        "area_ha"          : float — total changed area in hectares.
        "area_km2"         : float — total changed area in square kilometres.
        "n_changed_pixels" : int
        "total_pixels"     : int
        "pct_changed"      : float — percentage of image pixels that changed.
    """
    n_changed = int(mask.sum())
    total_px  = mask.size

    # Pixel ground sampling distance in map units (convert degrees to metres if in geographic CRS)
    gsd_x = abs(float(transform.a))
    gsd_y = abs(float(transform.e))
    if gsd_x < 0.05:  # Geographic degrees (e.g. EPSG:4326)
        _M_PER_DEG = 111_319.0
        pixel_area_m2 = (gsd_x * _M_PER_DEG) * (gsd_y * _M_PER_DEG)
    else:
        pixel_area_m2 = gsd_x * gsd_y

    area_m2  = n_changed * pixel_area_m2
    area_ha  = area_m2 / 10_000.0
    area_km2 = area_m2 / 1_000_000.0

    return {
        "area_m2":          round(area_m2, 2),
        "area_ha":          round(area_ha, 4),
        "area_km2":         round(area_km2, 6),
        "n_changed_pixels": n_changed,
        "total_pixels":     total_px,
        "pct_changed":      round(100.0 * n_changed / max(total_px, 1), 4),
    }


# ---------------------------------------------------------------------------
# Direction classification
# ---------------------------------------------------------------------------

def classify_change_direction(
    signed_diff_primary: np.ndarray,
    change_mask: np.ndarray,
    primary_index: str = "ndvi",
    signed_diff_secondary: np.ndarray | None = None,
    secondary_index: str | None = None,
) -> ChangeLabel:
    """
    Classify the dominant direction of change over the changed pixels.

    Parameters
    ----------
    signed_diff_primary   : (H, W) float32 signed difference for the primary index
                            (T2 - T1 > 0 means the index increased).
    change_mask           : (H, W) bool — which pixels changed.
    primary_index         : Index name ("ndvi", "ndwi", "ndbi", "rvi", …).
    signed_diff_secondary : Optional secondary index signed difference.
    secondary_index       : Name of the secondary index.

    Returns
    -------
    ChangeLabel string.
    """
    pct_changed = change_mask.sum() / max(change_mask.size, 1)
    if pct_changed < _MIN_CHANGED_FRACTION:
        return "no_change"

    # Mean signed change within the change mask
    changed_pixels = signed_diff_primary[change_mask]
    if changed_pixels.size == 0:
        return "no_change"

    mean_delta = float(np.median(changed_pixels))  # median is more robust than mean

    label = _classify_single_index(primary_index, mean_delta)

    # Cross-check with secondary index if available
    if signed_diff_secondary is not None and secondary_index is not None:
        sec_pixels   = signed_diff_secondary[change_mask]
        sec_mean     = float(np.median(sec_pixels))
        sec_label    = _classify_single_index(secondary_index, sec_mean)

        if sec_label != label and sec_label != "no_change":
            # Contradicting signals — flag as mixed
            logger.debug(
                "Mixed change: primary=%s (%s) vs secondary=%s (%s)",
                primary_index, label, secondary_index, sec_label,
            )
            return "mixed_change"

    return label


def _classify_single_index(index_name: str, mean_delta: float) -> ChangeLabel:
    """Map (index, direction) to a semantic ChangeLabel."""
    idx = index_name.lower()
    delta_threshold = 0.02  # minimum mean delta to claim directional change

    if abs(mean_delta) < delta_threshold:
        return "no_change"

    if idx == "ndvi":
        return "vegetation_gain" if mean_delta > 0 else "vegetation_loss"

    if idx == "ndwi":
        return "water_expansion" if mean_delta > 0 else "water_contraction"

    if idx == "ndbi":
        # NDBI increase → more built-up surface
        return "urban_growth" if mean_delta > 0 else "vegetation_gain"

    if idx in ("rvi", "db_vv", "db_vh"):
        return "sar_change"

    return "mixed_change"


# ---------------------------------------------------------------------------
# Natural-language summary builder
# ---------------------------------------------------------------------------

_DIRECTION_DESCRIPTIONS: dict[str, str] = {
    "vegetation_loss":    "significant vegetation loss was detected",
    "vegetation_gain":    "significant vegetation gain (or regrowth) was detected",
    "water_expansion":    "a substantial expansion of surface water was detected",
    "water_contraction":  "a significant reduction in surface water extent was detected",
    "urban_growth":       "new built-up or impervious surface development was detected",
    "sar_change":         "a change in SAR backscatter consistent with ground-surface alteration was detected",
    "mixed_change":       "conflicting spectral signals suggest mixed land-cover change",
    "no_change":          "no significant change was detected within the analysed area",
}


def build_text_summary(
    area_metrics: dict[str, float],
    direction: ChangeLabel,
    primary_index: str,
    n_regions: int,
    timestamp_t1: str | None = None,
    timestamp_t2: str | None = None,
    n_pseudo_removed: int = 0,
    confidence: float = 0.0,
) -> str:
    """
    Construct a plain-English summary paragraph suitable for the API response
    and the non-expert user UX.

    Returns
    -------
    Multi-sentence summary string.
    """
    direction_desc = _DIRECTION_DESCRIPTIONS.get(direction, "change was detected")
    area_ha = area_metrics.get("area_ha", 0.0)
    pct = area_metrics.get("pct_changed", 0.0)
    n_px = area_metrics.get("n_changed_pixels", 0)

    # Temporal clause
    if timestamp_t1 and timestamp_t2:
        temporal_clause = f"Between {timestamp_t1} and {timestamp_t2}, "
    else:
        temporal_clause = "Between the two input images, "

    # Area clause
    if area_ha >= 1.0:
        area_clause = f"{area_ha:.2f} ha ({pct:.1f}% of the image)"
    else:
        area_clause = f"{int(n_px):,} pixels ({pct:.2f}% of the image)"

    # Region clause
    region_clause = (
        f"spanning {n_regions} distinct region{'s' if n_regions != 1 else ''}"
    )

    # Confidence clause
    conf_clause = f" Confidence score: {confidence:.2f}/1.00."

    # Pseudo-change note
    pseudo_note = ""
    if n_pseudo_removed > 0:
        pseudo_note = (
            f" ({n_pseudo_removed:,} pseudo-change pixels from radiometric drift "
            f"were suppressed before thresholding.)"
        )

    summary = (
        f"{temporal_clause}{direction_desc}, covering approximately {area_clause}, "
        f"{region_clause}. The primary spectral indicator used was "
        f"{primary_index.upper()}.{conf_clause}{pseudo_note}"
    )

    return summary
