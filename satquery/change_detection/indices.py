"""
satquery.change_detection.indices
==================================
Spectral index computation for optical and SAR imagery.

Optical indices
---------------
- NDVI  — Normalized Difference Vegetation Index     (NIR - R) / (NIR + R)
- NDWI  — Normalized Difference Water Index          (G - NIR) / (G + NIR)
- NDBI  — Normalized Difference Built-up Index       (SWIR - NIR) / (SWIR + NIR)
- EVI   — Enhanced Vegetation Index (three-band)

SAR indices
-----------
- Backscatter ratio  (VV / VH)   — sensitive to surface roughness change
- RVI     — Radar Vegetation Index  4·VH / (VV + VH)
- dB      — convert linear amplitude to dB scale

Band layout conventions (0-indexed)
------------------------------------
Optical (Sentinel-2 / Cartosat-like multispectral):
    0 = Blue, 1 = Green, 2 = Red, 3 = NIR, 4 = SWIR

SAR (RISAT-1C / Sentinel-1 dual-pol):
    0 = VV (or HH), 1 = VH (or HV)

All functions accept (H, W) float32 numpy arrays and return (H, W) float32.
Division by zero is silenced (returns 0.0 at degenerate pixels).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-10   # guard for division by zero


# ---------------------------------------------------------------------------
# Generic safe-division helper
# ---------------------------------------------------------------------------

def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(A - B) / (A + B) clamped to [-1, 1] with zero-safe denominator."""
    num = a.astype(np.float32) - b.astype(np.float32)
    den = a.astype(np.float32) + b.astype(np.float32)
    return np.where(np.abs(den) < _EPS, 0.0, num / den).astype(np.float32)


# ---------------------------------------------------------------------------
# Optical spectral indices
# ---------------------------------------------------------------------------

def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.
    Range: [-1, 1].  Dense vegetation ≈ 0.5–0.9.
    """
    return _norm_diff(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index (McFeeters 1996).
    Range: [-1, 1].  Open water ≈ 0.3–1.0.
    """
    return _norm_diff(green, nir)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Built-up Index (Zha 2003).
    Range: [-1, 1].  Built-up surfaces ≈ 0.1–0.5.
    """
    return _norm_diff(swir, nir)


def evi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> np.ndarray:
    """
    Enhanced Vegetation Index (Huete et al. 2002).
    More robust than NDVI in high-biomass / atmospheric aerosol conditions.
    """
    b, r, n = blue.astype(np.float32), red.astype(np.float32), nir.astype(np.float32)
    den = n + C1 * r - C2 * b + L
    result = np.where(np.abs(den) < _EPS, 0.0, G * (n - r) / den)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# SAR indices
# ---------------------------------------------------------------------------

def sar_amplitude_to_db(amplitude: np.ndarray) -> np.ndarray:
    """
    Convert SAR linear amplitude (power) to decibel scale.
    σ° [dB] = 10 · log₁₀(max(amplitude, ε))
    """
    amp = np.maximum(amplitude.astype(np.float32), _EPS)
    return (10.0 * np.log10(amp)).astype(np.float32)


def sar_backscatter_ratio(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    VV/VH backscatter ratio (linear amplitude).
    Sensitive to surface roughness and dielectric properties.
    """
    return np.where(np.abs(vh) < _EPS, 0.0, vv.astype(np.float32) / np.maximum(vh, _EPS)).astype(np.float32)


def rvi(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Radar Vegetation Index: 4·VH / (VV + VH).
    Range: [0, 1].  Dense crop/forest ≈ 0.6–1.0.
    """
    num = 4.0 * vh.astype(np.float32)
    den = vv.astype(np.float32) + vh.astype(np.float32)
    return np.where(np.abs(den) < _EPS, 0.0, num / den).astype(np.float32)


# ---------------------------------------------------------------------------
# Band extraction helper
# ---------------------------------------------------------------------------

def extract_index(
    array: np.ndarray,
    index_name: str,
    band_map: dict[str, int] | None = None,
) -> np.ndarray | None:
    """
    Compute a named spectral index from a multi-band (B, H, W) float32 array.

    Parameters
    ----------
    array      : (B, H, W) float32.
    index_name : One of {"ndvi", "ndwi", "ndbi", "evi", "rvi", "db_vv", "db_vh"}.
    band_map   : Maps band role → band index (0-based).
                 Defaults to Sentinel-2 / Cartosat-like layout for optical
                 and RISAT-1C layout for SAR.

    Returns
    -------
    (H, W) float32 index array, or ``None`` if required bands are unavailable.
    """
    # Default band layouts
    _optical_default: dict[str, int] = {
        "blue": 0, "green": 1, "red": 2, "nir": 3, "swir": 4
    }
    _sar_default: dict[str, int] = {"vv": 0, "vh": 1}

    if hasattr(array, "array"):
        array = getattr(array, "array")

    n_bands = array.shape[0]
    bm = band_map or {}

    def _get(role: str, default_map: dict[str, int]) -> np.ndarray | None:
        idx = bm.get(role, default_map.get(role))
        if idx is None or idx >= n_bands:
            return None
        return array[idx]

    name = index_name.lower()

    if name == "ndvi":
        red, nir = _get("red", _optical_default), _get("nir", _optical_default)
        return ndvi(red, nir) if red is not None and nir is not None else None

    if name == "ndwi":
        green, nir = _get("green", _optical_default), _get("nir", _optical_default)
        return ndwi(green, nir) if green is not None and nir is not None else None

    if name == "ndbi":
        swir, nir = _get("swir", _optical_default), _get("nir", _optical_default)
        return ndbi(swir, nir) if swir is not None and nir is not None else None

    if name == "evi":
        blue = _get("blue", _optical_default)
        red  = _get("red",  _optical_default)
        nir  = _get("nir",  _optical_default)
        if any(x is None for x in [blue, red, nir]):
            return None
        return evi(blue, red, nir)

    if name == "rvi":
        vv, vh = _get("vv", _sar_default), _get("vh", _sar_default)
        return rvi(vv, vh) if vv is not None and vh is not None else None

    if name in ("db_vv", "db_vh"):
        role = "vv" if name == "db_vv" else "vh"
        amp = _get(role, _sar_default)
        return sar_amplitude_to_db(amp) if amp is not None else None

    raise ValueError(f"Unknown index_name: {index_name!r}")


# ---------------------------------------------------------------------------
# Available-index discovery
# ---------------------------------------------------------------------------

def available_indices(array: np.ndarray, sensor: str = "optical") -> list[str]:
    """
    Return the list of indices that can be computed from *array* given *sensor*.
    Used by the pipeline to auto-select the best index for a query.
    """
    n = array.shape[0]
    if sensor == "optical":
        candidates = ["ndvi", "ndwi", "ndbi", "evi"]
        # Minimum bands: ndvi=4, ndwi=4, ndbi=5, evi=4
        min_bands  = {"ndvi": 4, "ndwi": 4, "ndbi": 5, "evi": 4}
        return [idx for idx in candidates if n >= min_bands[idx]]
    if sensor == "sar":
        if n >= 2:
            return ["rvi", "db_vv", "db_vh"]
        return ["db_vv"]
    return []
