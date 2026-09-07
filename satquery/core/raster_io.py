"""
satquery.core.raster_io
=======================
GeoTIFF loading, reprojection, band extraction and GeoJSON export utilities.

Responsibilities
----------------
- Open GeoTIFF with rasterio and expose a normalised numpy array + metadata dict.
- Reproject / resample one image to match another's CRS, transform and shape.
- Write a binary change-mask back to GeoTIFF for archiving.
- Polygonize a binary mask and export as a GeoJSON FeatureCollection.

All public functions are stateless and accept / return plain Python objects
(numpy arrays, dicts, lists) so they are trivially testable without disk I/O.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model for a loaded raster
# ---------------------------------------------------------------------------

class RasterData:
    """
    Thin wrapper around a numpy array and rasterio metadata.

    Attributes
    ----------
    array : np.ndarray
        Shape (bands, height, width), float32.
    meta  : dict
        Rasterio profile dict (crs, transform, dtype, count, …).
    path  : str | None
        Source file path (informational only).
    """

    __slots__ = ("array", "meta", "path")

    def __init__(self, array: np.ndarray, meta: dict, path: str | None = None):
        self.array = array.astype(np.float32)
        self.meta = meta
        self.path = path

    # Convenience properties
    @property
    def bands(self) -> int:
        return int(self.array.shape[0])

    @property
    def height(self) -> int:
        return int(self.array.shape[1])

    @property
    def width(self) -> int:
        return int(self.array.shape[2])

    @property
    def crs(self) -> CRS:
        return self.meta.get("crs")

    @property
    def transform(self):
        return self.meta.get("transform")

    def band(self, index: int = 0) -> np.ndarray:
        """Return a single band (0-indexed) as a 2-D float32 array."""
        return self.array[index]

    def __repr__(self) -> str:
        crs_str = self.crs.to_epsg() if self.crs else "unknown"
        return (
            f"<RasterData bands={self.bands} h={self.height} w={self.width} "
            f"crs=EPSG:{crs_str} src={os.path.basename(self.path or 'array')}>"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raster(path: str, max_bands: int | None = None) -> RasterData:
    """
    Load a GeoTIFF from *path* and return a :class:`RasterData`.

    Parameters
    ----------
    path      : Absolute or relative path to a GeoTIFF file.
    max_bands : If given, only the first *max_bands* bands are read.

    Returns
    -------
    RasterData
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        n = src.count if max_bands is None else min(src.count, max_bands)
        band_indices = list(range(1, n + 1))
        array = src.read(band_indices).astype(np.float32)
        meta = src.profile.copy()
        meta["count"] = n

    logger.debug("Loaded raster %s → %s", path, array.shape)
    return RasterData(array=array, meta=meta, path=path)


def load_raster_from_array(
    array: np.ndarray,
    crs: str | CRS = "EPSG:4326",
    transform=None,
) -> RasterData:
    """
    Wrap a numpy array (H×W or B×H×W) as a :class:`RasterData` without disk I/O.
    Useful for synthetic test data and unit testing.
    """
    if array.ndim == 2:
        array = array[np.newaxis, ...]  # add bands dim
    import affine
    if transform is None:
        transform = affine.Affine(1.0, 0.0, 0.0, 0.0, -1.0, array.shape[1])
    if not isinstance(crs, CRS):
        crs = CRS.from_string(crs)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": array.shape[0],
        "height": array.shape[1],
        "width": array.shape[2],
        "crs": crs,
        "transform": transform,
    }
    return RasterData(array=array.astype(np.float32), meta=meta)


# ---------------------------------------------------------------------------
# Reprojection / co-registration
# ---------------------------------------------------------------------------

def match_raster(
    source: RasterData,
    reference: RasterData,
    resampling: Resampling = Resampling.bilinear,
) -> RasterData:
    """
    Reproject and resample *source* to match *reference* CRS, transform and shape.

    Returns a new :class:`RasterData`; the originals are not modified.
    """
    dst_crs = reference.crs
    dst_transform = reference.transform
    dst_height = reference.height
    dst_width = reference.width

    dst_array = np.zeros(
        (source.bands, dst_height, dst_width), dtype=np.float32
    )

    for b in range(source.bands):
        reproject(
            source=source.array[b],
            destination=dst_array[b],
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    new_meta = source.meta.copy()
    new_meta.update(
        {
            "crs": dst_crs,
            "transform": dst_transform,
            "height": dst_height,
            "width": dst_width,
        }
    )
    logger.debug("Reprojected raster to match reference → shape %s", dst_array.shape)
    return RasterData(array=dst_array, meta=new_meta, path=source.path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_raster(data: RasterData, path: str) -> str:
    """
    Write *data* to a GeoTIFF at *path*. Returns *path*.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    profile = data.meta.copy()
    profile.update(
        {
            "driver": "GTiff",
            "dtype": "float32",
            "count": data.bands,
            "height": data.height,
            "width": data.width,
            "compress": "lzw",
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.array)
    logger.info("Saved raster → %s", path)
    return path


def save_mask(
    mask: np.ndarray,
    reference: RasterData,
    path: str,
) -> str:
    """
    Save a boolean/uint8 2-D *mask* as a single-band GeoTIFF aligned to
    *reference*. Returns *path*.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    profile = reference.meta.copy()
    profile.update(
        {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "compress": "lzw",
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8)[np.newaxis, ...])
    logger.info("Saved mask → %s", path)
    return path


# ---------------------------------------------------------------------------
# Polygonization → GeoJSON
# ---------------------------------------------------------------------------

def mask_to_geojson(
    mask: np.ndarray,
    reference: RasterData,
    label: str = "change",
    min_area_px: int = 4,
) -> dict[str, Any]:
    """
    Convert a binary 2-D *mask* to a GeoJSON FeatureCollection.

    Parameters
    ----------
    mask         : (H, W) boolean or uint8 array.  1 = changed pixel.
    reference    : Used for CRS and spatial transform.
    label        : Attribute value written to each feature's ``class`` property.
    min_area_px  : Polygons smaller than this (in pixels) are discarded.

    Returns
    -------
    dict  — GeoJSON FeatureCollection (plain Python dict, JSON-serialisable).
    """
    mask_u8 = mask.astype(np.uint8)
    transform = reference.transform
    crs = reference.crs

    pixel_area = abs(float(transform.a * transform.e))
    if pixel_area < 1e-12:
        pixel_area = 1.0

    features = []
    for geom_dict, value in rasterio_shapes(mask_u8, transform=transform):
        if value == 0:
            continue
        geom = shape(geom_dict)
        approx_px = geom.area / pixel_area
        if approx_px < min_area_px:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "class": label,
                    "area_px": round(approx_px, 1),
                },
            }
        )

    crs_str = crs.to_epsg() if crs else None
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs_str}"},
        },
        "features": features,
    }


def save_geojson(geojson: dict, path: str) -> str:
    """Serialise *geojson* to *path* and return path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    logger.info("Saved GeoJSON → %s (%d features)", path, len(geojson.get("features", [])))
    return path
