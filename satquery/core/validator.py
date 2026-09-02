"""
satquery.core.validator
=======================
GeoValidator — spatial/geospatial consistency and sanity checks.

Every public function raises a :class:`ValidationError` (subclass of
``ValueError``) with a structured message if a check fails. All checks are
intentionally cheap (no heavy image processing) so they can run before the
main pipeline to fail fast and produce clear error messages.

Checks implemented
------------------
1. File existence
2. CRS compatibility (both in same coordinate system family)
3. Spatial overlap  (bounding boxes must intersect)
4. Resolution consistency (warn if >10× scale mismatch)
5. Band count validity for a given sensor/modality type
6. Timestamp ordering  (T1 must precede T2)
7. Polygon-in-image-bounds (for GeoJSON outputs, post-processing)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when a geospatial validation check fails."""

    def __init__(self, check: str, detail: str):
        self.check = check
        self.detail = detail
        super().__init__(f"[{check}] {detail}")

    def to_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------

def check_file_exists(path: str) -> None:
    """Raise :class:`ValidationError` if *path* does not exist or is empty."""
    if not os.path.isfile(path):
        raise ValidationError("FILE_EXISTS", f"File not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValidationError("FILE_EXISTS", f"File is empty (0 bytes): {path}")


def check_raster_readable(path: str) -> dict[str, Any]:
    """
    Open *path* with rasterio and return a summary dict.
    Raises :class:`ValidationError` if the file cannot be opened.
    """
    check_file_exists(path)
    try:
        with rasterio.open(path) as src:
            info = {
                "path": path,
                "crs": src.crs.to_string() if src.crs else None,
                "transform": src.transform,
                "count": src.count,
                "height": src.height,
                "width": src.width,
                "bounds": src.bounds,
                "dtype": src.dtypes[0],
            }
    except Exception as exc:
        raise ValidationError("RASTER_READABLE", f"Cannot open raster {path}: {exc}") from exc
    return info


# ---------------------------------------------------------------------------
# Pair-level spatial checks
# ---------------------------------------------------------------------------

def check_crs_compatible(info_t1: dict, info_t2: dict) -> None:
    """
    Warn if the two rasters have different CRS strings.
    Raises if one or both have no CRS at all.
    """
    crs1 = info_t1.get("crs")
    crs2 = info_t2.get("crs")
    if not crs1 or not crs2:
        raise ValidationError(
            "CRS_COMPATIBLE",
            f"One or both rasters lack a CRS (T1={crs1!r}, T2={crs2!r}). "
            "Ensure inputs are projected or geographic GeoTIFFs.",
        )
    if crs1 != crs2:
        logger.warning(
            "CRS mismatch: T1=%s vs T2=%s — co-registration will be applied.", crs1, crs2
        )


def check_spatial_overlap(info_t1: dict, info_t2: dict) -> None:
    """
    Raise :class:`ValidationError` if the bounding boxes of T1 and T2 do not
    spatially overlap (i.e., there is no valid area of interest).

    NOTE: Assumes both images share a CRS (or have already been matched).
    """
    b1 = info_t1["bounds"]   # rasterio.coords.BoundingBox
    b2 = info_t2["bounds"]

    x_overlap = b1.left < b2.right and b2.left < b1.right
    y_overlap = b1.bottom < b2.top  and b2.bottom < b1.top

    if not (x_overlap and y_overlap):
        raise ValidationError(
            "SPATIAL_OVERLAP",
            f"T1 bounds {b1} and T2 bounds {b2} do not overlap. "
            "Ensure both images cover the same geographic area.",
        )


def check_resolution_consistency(
    info_t1: dict,
    info_t2: dict,
    tolerance_factor: float = 10.0,
) -> None:
    """
    Log a warning if the ground sample distance (GSD) of T1 and T2 differ by
    more than *tolerance_factor*×. Does not raise — only warns, because
    the pipeline will resample before differencing.
    """
    tr1 = info_t1["transform"]
    tr2 = info_t2["transform"]
    gsd1 = abs(tr1.a)
    gsd2 = abs(tr2.a)
    if gsd1 == 0 or gsd2 == 0:
        return
    ratio = max(gsd1, gsd2) / min(gsd1, gsd2)
    if ratio > tolerance_factor:
        logger.warning(
            "Large resolution mismatch: GSD T1=%.4f vs T2=%.4f (ratio=%.1f×). "
            "Significant resampling artefacts may appear in the difference map.",
            gsd1, gsd2, ratio,
        )


def check_band_count(
    info: dict,
    sensor_type: str = "optical",
    min_bands: int = 1,
) -> None:
    """
    Validate band count for a given *sensor_type*.

    Accepted sensor_type values:
    - "optical"   : ≥ 3 bands expected (RGB or multispectral)
    - "sar"       : 1 or 2 bands expected (amplitude, ±phase)
    - "any"       : only checks min_bands
    """
    count = info["count"]
    path = info["path"]

    if sensor_type == "optical" and count < 3:
        logger.warning(
            "Optical raster %s has only %d band(s). "
            "NDVI/NDWI/NDBI require Near-IR and SWIR bands beyond RGB. "
            "Indices unavailable for this input.",
            path, count,
        )
    elif sensor_type == "sar" and count > 2:
        logger.warning(
            "SAR raster %s has %d bands — expected 1 (amplitude) or 2 (VV+VH). "
            "Only the first 2 bands will be used.",
            path, count,
        )

    if count < min_bands:
        raise ValidationError(
            "BAND_COUNT",
            f"Raster {path} has {count} band(s), minimum required is {min_bands}.",
        )


# ---------------------------------------------------------------------------
# Temporal check
# ---------------------------------------------------------------------------

def check_temporal_order(
    timestamp_t1: str | datetime | None,
    timestamp_t2: str | datetime | None,
) -> None:
    """
    Raise :class:`ValidationError` if T1 is not before T2.
    Silently passes if either timestamp is ``None`` (unknown).
    Accepts ISO-8601 strings or :class:`datetime` objects.
    """
    if timestamp_t1 is None or timestamp_t2 is None:
        logger.debug("Temporal order check skipped — one or both timestamps missing.")
        return

    def _parse(ts: str | datetime) -> datetime:
        if isinstance(ts, datetime):
            return ts
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        raise ValidationError("TEMPORAL_ORDER", f"Cannot parse timestamp: {ts!r}")

    dt1 = _parse(timestamp_t1)
    dt2 = _parse(timestamp_t2)
    if dt1 >= dt2:
        raise ValidationError(
            "TEMPORAL_ORDER",
            f"T1 ({dt1.date()}) must precede T2 ({dt2.date()}). "
            "Swap the input images and retry.",
        )


# ---------------------------------------------------------------------------
# Post-processing checks (GeoValidator)
# ---------------------------------------------------------------------------

def check_polygons_in_bounds(
    geojson: dict,
    bounds: tuple[float, float, float, float],  # (minx, miny, maxx, maxy)
    tolerance: float = 1e-6,
) -> list[str]:
    """
    Inspect every feature in *geojson* and collect warnings for polygons whose
    centroid lies outside *bounds*.

    Returns a list of warning strings (empty list = all OK).
    This does NOT raise — callers decide whether to filter or just log warnings.
    """
    from shapely.geometry import shape as _shape, box as _box

    minx, miny, maxx, maxy = bounds
    image_box = _box(
        minx - tolerance, miny - tolerance, maxx + tolerance, maxy + tolerance
    )
    warnings: list[str] = []

    for i, feat in enumerate(geojson.get("features", [])):
        try:
            geom = _shape(feat["geometry"])
            if not image_box.contains(geom.centroid):
                warnings.append(
                    f"Feature {i}: centroid {geom.centroid.coords[0]} is outside "
                    f"image bounds {bounds}. Possible CRS mismatch or artefact."
                )
        except Exception as exc:
            warnings.append(f"Feature {i}: cannot parse geometry — {exc}")

    if warnings:
        for w in warnings:
            logger.warning("[GeoValidator] %s", w)

    return warnings


# ---------------------------------------------------------------------------
# Convenience: run all pair checks at once
# ---------------------------------------------------------------------------

def validate_input_pair(
    path_t1: str,
    path_t2: str,
    sensor_t1: str = "optical",
    sensor_t2: str = "optical",
    timestamp_t1: str | datetime | None = None,
    timestamp_t2: str | datetime | None = None,
) -> tuple[dict, dict]:
    """
    Run the full suite of input validation checks on a T1/T2 pair.

    Returns
    -------
    (info_t1, info_t2) — rasterio summary dicts on success.

    Raises
    ------
    :class:`ValidationError` on any hard failure.
    """
    info_t1 = check_raster_readable(path_t1)
    info_t2 = check_raster_readable(path_t2)
    check_crs_compatible(info_t1, info_t2)
    check_spatial_overlap(info_t1, info_t2)
    check_resolution_consistency(info_t1, info_t2)
    check_band_count(info_t1, sensor_type=sensor_t1)
    check_band_count(info_t2, sensor_type=sensor_t2)
    check_temporal_order(timestamp_t1, timestamp_t2)

    logger.info(
        "Input pair validation passed: %s (T1) | %s (T2)",
        os.path.basename(path_t1),
        os.path.basename(path_t2),
    )
    return info_t1, info_t2
