"""
satquery.change_detection.confidence
=======================================
Bimodal histogram separation confidence scoring for the Otsu change mask.

Theory
------
Otsu's method works best when the difference map has a bimodal histogram
(two distinct peaks: "no-change" and "change").  If the histogram is
unimodal or nearly flat, Otsu still produces a threshold but it separates
poorly — the resulting mask is unreliable.

This module quantifies that separation quality as a confidence score [0, 1]
using three complementary measures:

  1. **Otsu inter-class variance ratio** (ω)
     Ratio of Otsu inter-class variance to total variance.
     Range [0, 1]; higher = better bimodal separation.

  2. **Valley-to-peak ratio** (v)
     Depth of the valley between the two histogram modes relative to the
     taller peak.  Values near 0 indicate a well-defined valley (confident).

  3. **Imbalance penalty** (p)
     A hard penalty if the change fraction is < 0.5% or > 60% of the image,
     which are both physically implausible (tiny artefacts or saturated masks).

Final confidence = ω × (1 - v) × (1 - p), clamped to [0, 1].

Usage
-----
This score is surfaced in the API response under ``"confidence"`` and
displayed as a badge in the Frontend evidence drawer (Research Gap #3 —
Cross-modal confidence discounting).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


def compute_confidence(
    diff: np.ndarray,
    change_mask: np.ndarray,
    otsu_threshold: float,
    n_bins: int = 256,
) -> float:
    """
    Compute a bimodal histogram separation confidence score.

    Parameters
    ----------
    diff           : Suppressed/smoothed difference map (H, W) float32.
    change_mask    : (H, W) bool — the final binary change mask.
    otsu_threshold : The threshold value chosen by Otsu.
    n_bins         : Number of histogram bins.

    Returns
    -------
    float in [0.0, 1.0].
    """
    flat = diff.flatten().astype(np.float64)
    n_px = flat.size

    if flat.max() - flat.min() < 1e-8:
        logger.debug("Uniform diff map — confidence = 0.0")
        return 0.0

    # ----- 1. Otsu inter-class variance ratio -----
    omega = _otsu_variance_ratio(flat, otsu_threshold)

    # ----- 2. Valley-to-peak ratio -----
    hist, bin_edges = np.histogram(flat, bins=n_bins, density=False)
    hist_smooth = np.convolve(hist.astype(float), np.ones(5) / 5, mode="same")
    valley_score = _valley_score(hist_smooth, bin_edges, otsu_threshold)

    # ----- 3. Imbalance penalty -----
    changed_frac = change_mask.sum() / max(n_px, 1)
    imbalance = _imbalance_penalty(changed_frac)

    # ----- Combine -----
    confidence = float(omega * (1.0 - valley_score) * (1.0 - imbalance))
    confidence = float(np.clip(confidence, 0.0, 1.0))

    logger.debug(
        "Confidence breakdown: ω=%.3f, valley=%.3f, imbalance=%.3f → final=%.3f",
        omega, valley_score, imbalance, confidence,
    )
    return round(confidence, 4)


# ---------------------------------------------------------------------------
# Sub-components
# ---------------------------------------------------------------------------

def _otsu_variance_ratio(flat: np.ndarray, threshold: float) -> float:
    """
    Return the Otsu inter-class variance / total variance ratio.
    Clamped to [0, 1].
    """
    below = flat[flat <= threshold]
    above = flat[flat > threshold]

    if below.size == 0 or above.size == 0:
        return 0.0

    w0 = len(below) / len(flat)
    w1 = len(above) / len(flat)
    mu0 = below.mean()
    mu1 = above.mean()
    mu  = flat.mean()

    inter_var = w0 * (mu0 - mu) ** 2 + w1 * (mu1 - mu) ** 2
    total_var = float(flat.var())

    if total_var < 1e-12:
        return 0.0
    return float(np.clip(inter_var / total_var, 0.0, 1.0))


def _valley_score(
    hist_smooth: np.ndarray,
    bin_edges: np.ndarray,
    threshold: float,
) -> float:
    """
    Measure how deep the valley between the two modes is.
    Returns 0 if the histogram is well-separated (good), 1 if flat (bad).
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(hist_smooth)

    # Find threshold bin
    thresh_bin = int(np.searchsorted(bin_centers, threshold, side="right"))
    thresh_bin = max(1, min(thresh_bin, n_bins - 2))

    # Peak heights on each side
    left_peak  = float(hist_smooth[:thresh_bin].max()) if thresh_bin > 0 else 0.0
    right_peak = float(hist_smooth[thresh_bin:].max()) if thresh_bin < n_bins else 0.0
    valley_val = float(hist_smooth[thresh_bin])

    taller_peak = max(left_peak, right_peak)
    if taller_peak < 1e-8:
        return 1.0   # flat histogram = no confidence

    # Ratio of valley height to taller peak: 0 = perfect valley, 1 = no dip
    ratio = valley_val / taller_peak
    return float(np.clip(ratio, 0.0, 1.0))


def _imbalance_penalty(changed_frac: float) -> float:
    """
    Penalise physically implausible change fractions.
    Returns 0 (no penalty) for 0.5%–60%, linearly ramps to 1 outside.
    """
    LOW  = 0.005   # 0.5%
    HIGH = 0.60    # 60%

    if LOW <= changed_frac <= HIGH:
        return 0.0
    if changed_frac < LOW:
        # Smoothly ramp: 0 at LOW, 1 at 0
        return float(np.clip(1.0 - changed_frac / LOW, 0.0, 1.0))
    else:
        # Smoothly ramp: 0 at HIGH, 1 at 1.0
        return float(np.clip((changed_frac - HIGH) / (1.0 - HIGH), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Confidence label helper
# ---------------------------------------------------------------------------

def confidence_label(score: float) -> str:
    """Convert a numeric confidence score to a human-readable label."""
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    if score >= 0.20:
        return "LOW"
    return "VERY_LOW"
