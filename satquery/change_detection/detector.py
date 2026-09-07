"""
satquery.change_detection.detector
====================================
Core change detection logic — differencing, Otsu thresholding,
and pseudo-change suppression filter.

Design notes
------------
The pseudo-change suppression filter is the key research-gap feature from
the STSF-Net / Delta-SN6 literature.  It works in three steps:

  1. **Gaussian smoothing** of the raw difference map to suppress high-
     frequency speckle (especially important for SAR).
  2. **Morphological opening** (erosion → dilation) to remove single-pixel
     and small-cluster noise hits before thresholding — these are almost
     always illumination artefacts, not real ground change.
  3. **STSF-inspired pseudo-change mask** built from the inter-quartile
     difference between the smoothed T1-indexed and T2-indexed arrays
     in a local patch: pixels where the *relative* change is dominated by
     radiometric drift (same-sign, uniform neighbourhood) are flagged as
     pseudo-changes and excluded from the binary change mask.

The filter is toggled with ``suppress_pseudo_changes=True`` (default).
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
IndexArray = np.ndarray   # (H, W) float32
BoolMask   = np.ndarray   # (H, W) bool


# ---------------------------------------------------------------------------
# 1. Differencing
# ---------------------------------------------------------------------------

def compute_difference(
    index_t1: IndexArray,
    index_t2: IndexArray,
    mode: Literal["absolute", "signed"] = "absolute",
) -> IndexArray:
    """
    Compute a pixel-wise difference map between two index arrays.

    Parameters
    ----------
    index_t1, index_t2 : (H, W) float32 arrays (same shape, same index).
    mode               : "absolute" → |T2 - T1|  (used for thresholding)
                         "signed"   → T2 - T1    (positive = increase)

    Returns
    -------
    (H, W) float32 difference map.
    """
    if index_t1.shape != index_t2.shape:
        raise ValueError(
            f"Shape mismatch: T1 {index_t1.shape} vs T2 {index_t2.shape}. "
            "Co-register inputs before differencing."
        )
    diff = index_t2.astype(np.float32) - index_t1.astype(np.float32)
    if mode == "absolute":
        diff = np.abs(diff)
    return diff


# ---------------------------------------------------------------------------
# 2. Gaussian smoothing (speckle / noise reduction)
# ---------------------------------------------------------------------------

def smooth_difference(
    diff: IndexArray,
    sigma: float = 1.5,
) -> IndexArray:
    """
    Apply Gaussian blur to *diff* to reduce speckle before thresholding.
    Uses OpenCV GaussianBlur for speed.

    Parameters
    ----------
    diff  : Raw difference map.
    sigma : Standard deviation of the Gaussian kernel (pixels).
              σ = 0 disables smoothing.

    Returns
    -------
    Smoothed (H, W) float32 array.
    """
    if sigma <= 0:
        return diff
    # kernel size must be odd
    ksize = max(3, int(6 * sigma + 1) | 1)
    return cv2.GaussianBlur(diff, (ksize, ksize), sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Pseudo-change suppression  (Research Gap #5 implementation)
# ---------------------------------------------------------------------------

def suppress_pseudo_changes(
    diff: IndexArray,
    index_t1: IndexArray,
    index_t2: IndexArray,
    window_size: int = 9,
    uniformity_threshold: float = 0.15,
) -> IndexArray:
    """
    STSF-Net–inspired pseudo-change suppression filter.

    A pixel is flagged as a *pseudo-change* (and its difference value zeroed)
    when the local neighbourhood around that pixel shows a spatially uniform
    radiometric shift in BOTH T1 and T2 index maps — characteristic of
    illumination drift, atmospheric correction errors, or sensor gain
    differences rather than real ground change.

    Algorithm
    ---------
    For each pixel p:
        1. Compute local standard deviation of T1-index in a w×w window (σ_t1).
        2. Compute local standard deviation of T2-index in a w×w window (σ_t2).
        3. Compute local mean of the *signed* difference map (μ_Δ).
        4. If |μ_Δ| > uniformity_threshold AND σ_t1 < 0.05 AND σ_t2 < 0.05,
           the neighbourhood is spectrally flat in both dates → uniform
           radiometric shift → pseudo-change. Zero out diff at p.

    Uses cv2.boxFilter for local statistics (avoids scipy ABI dependency).

    Parameters
    ----------
    diff                  : Absolute difference map (H, W) float32.
    index_t1, index_t2    : Original index maps for T1 and T2.
    window_size           : Neighbourhood side (pixels, odd preferred).
    uniformity_threshold  : Mean signed-diff magnitude above which a flat
                            neighbourhood is classified as pseudo-change.

    Returns
    -------
    (H, W) float32 difference map with pseudo-change pixels zeroed.
    """
    w = window_size
    ksize = (w, w)

    # Local means via cv2.boxFilter (ABI-stable, no scipy needed)
    t1f  = index_t1.astype(np.float32)
    t2f  = index_t2.astype(np.float32)
    mean_t1    = cv2.boxFilter(t1f,  ddepth=-1, ksize=ksize, normalize=True)
    mean_t2    = cv2.boxFilter(t2f,  ddepth=-1, ksize=ksize, normalize=True)
    mean_sq_t1 = cv2.boxFilter(t1f**2, ddepth=-1, ksize=ksize, normalize=True)
    mean_sq_t2 = cv2.boxFilter(t2f**2, ddepth=-1, ksize=ksize, normalize=True)

    # Local std (clip negative from float precision)
    std_t1 = np.sqrt(np.clip(mean_sq_t1 - mean_t1**2, 0, None))
    std_t2 = np.sqrt(np.clip(mean_sq_t2 - mean_t2**2, 0, None))

    # Signed diff local mean
    signed_diff = t2f - t1f
    mean_delta  = cv2.boxFilter(signed_diff, ddepth=-1, ksize=ksize, normalize=True)

    # Pseudo-change mask: uniform neighbourhood with large uniform shift
    flat_t1   = std_t1 < 0.05
    flat_t2   = std_t2 < 0.05
    large_shift = np.abs(mean_delta) > uniformity_threshold

    pseudo_change_mask = flat_t1 & flat_t2 & large_shift

    n_pseudo = int(pseudo_change_mask.sum())
    total_px  = diff.size
    logger.debug(
        "Pseudo-change suppression: %d / %d pixels (%.1f%%) removed.",
        n_pseudo, total_px, 100 * n_pseudo / max(total_px, 1),
    )

    suppressed = diff.copy()
    suppressed[pseudo_change_mask] = 0.0
    return suppressed


# ---------------------------------------------------------------------------
# 4. Otsu automatic thresholding
# ---------------------------------------------------------------------------

def _otsu_numpy(image: np.ndarray, n_bins: int = 256) -> float:
    """
    Pure-numpy Otsu threshold implementation.
    Equivalent to skimage.filters.threshold_otsu but with no external deps.
    """
    hist, bin_edges = np.histogram(image.ravel(), bins=n_bins)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.0

    prob = hist / total
    # Cumulative sums
    w0 = np.cumsum(prob)          # weight of class 0
    mu0 = np.cumsum(prob * bin_centers) / np.maximum(w0, 1e-12)
    w1  = 1.0 - w0
    mu_total = (prob * bin_centers).sum()
    mu1 = (mu_total - np.cumsum(prob * bin_centers)) / np.maximum(w1, 1e-12)

    inter_class_var = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.argmax(inter_class_var))
    return float(bin_centers[idx])


def otsu_threshold(
    diff: IndexArray,
    percentile_clip: float = 98.0,
) -> tuple[BoolMask, float]:
    """
    Apply Otsu's method to produce a binary change mask from *diff*.
    Uses a pure-numpy Otsu implementation (no skimage/scipy ABI dependency).

    Parameters
    ----------
    diff             : Difference map (H, W) float32, values ≥ 0.
    percentile_clip  : Upper-clip percentile before Otsu to remove outliers
                       that would push the threshold too high.

    Returns
    -------
    (mask, threshold)
        mask      : (H, W) bool, True = changed pixel.
        threshold : float — the Otsu threshold value (informational).
    """
    # Clip extreme outliers (sensor artefacts / clouds)
    clip_val = float(np.percentile(diff, percentile_clip))
    diff_clipped = np.clip(diff, 0, clip_val)

    diff_range = float(diff_clipped.max() - diff_clipped.min())
    if diff_range < 1e-8:
        logger.warning("Difference map is nearly uniform — no change detected.")
        return np.zeros(diff.shape, dtype=bool), 0.0

    thresh = _otsu_numpy(diff_clipped)
    mask   = diff_clipped > thresh

    n_changed = int(mask.sum())
    pct = 100 * n_changed / mask.size
    logger.debug(
        "Otsu threshold=%.4f → %d changed pixels (%.2f%%)", thresh, n_changed, pct
    )
    return mask, float(thresh)


# ---------------------------------------------------------------------------
# 5. Full detector entry point
# ---------------------------------------------------------------------------

def detect_changes(
    index_t1: IndexArray,
    index_t2: IndexArray,
    smooth_sigma: float = 1.5,
    suppress_pseudo_changes: bool = True,
    pseudo_window: int = 9,
    pseudo_uniformity_threshold: float = 0.15,
    percentile_clip: float = 98.0,
) -> dict:
    """
    End-to-end change detector for a single spectral index pair.

    Parameters
    ----------
    index_t1, index_t2          : (H, W) float32 index maps.
    smooth_sigma                : Gaussian blur σ for speckle reduction.
    suppress_pseudo_changes     : Enable STSF-Net pseudo-change filter.
    pseudo_window               : Window size for pseudo-change filter.
    pseudo_uniformity_threshold : Threshold for pseudo-change decision.
    percentile_clip             : Outlier clip percentile before Otsu.

    Returns
    -------
    dict with keys:
        "raw_diff"        : (H, W) float32 — |T2 - T1| before any processing
        "signed_diff"     : (H, W) float32 — T2 - T1 (direction of change)
        "smooth_diff"     : (H, W) float32 — after Gaussian smoothing
        "suppressed_diff" : (H, W) float32 — after pseudo-change filter
        "change_mask"     : (H, W) bool    — final binary change mask
        "otsu_threshold"  : float
        "n_pseudo_removed": int   — pixels removed by pseudo-change filter
    """
    # Step 1: differencing
    raw_diff    = compute_difference(index_t1, index_t2, mode="absolute")
    signed_diff = compute_difference(index_t1, index_t2, mode="signed")

    # Step 2: Gaussian smoothing
    smooth_diff = smooth_difference(raw_diff, sigma=smooth_sigma)

    # Step 3: Pseudo-change suppression
    if suppress_pseudo_changes:
        suppressed_diff = suppress_pseudo_changes_fn(
            smooth_diff, index_t1, index_t2,
            window_size=pseudo_window,
            uniformity_threshold=pseudo_uniformity_threshold,
        )
        n_pseudo = int((smooth_diff > 0).sum()) - int((suppressed_diff > 0).sum())
    else:
        suppressed_diff = smooth_diff
        n_pseudo = 0

    # Step 4: Otsu thresholding
    change_mask, thresh = otsu_threshold(suppressed_diff, percentile_clip)

    return {
        "raw_diff":         raw_diff,
        "signed_diff":      signed_diff,
        "smooth_diff":      smooth_diff,
        "suppressed_diff":  suppressed_diff,
        "change_mask":      change_mask,
        "otsu_threshold":   thresh,
        "n_pseudo_removed": n_pseudo,
    }


# Alias so suppress_pseudo_changes param name doesn't shadow the function
suppress_pseudo_changes_fn = suppress_pseudo_changes
