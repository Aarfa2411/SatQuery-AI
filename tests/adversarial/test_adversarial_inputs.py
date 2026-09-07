"""
Adversarial & Input Robustness Tests (Workstream H)
Validates that GeoValidator and pipeline gracefully trap:
1. Inverted temporal order (T1 > T2).
2. Non-overlapping geographic bounds.
3. Missing CRS (CRS=None).
4. Corrupted raster files.
"""

import os
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from satquery.core.validator import (
    validate_input_pair,
    ValidationError,
    check_raster_readable,
    check_crs_compatible,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def create_dummy_raster(path: str, crs="EPSG:32643", origin=(500000, 1400000), shape=(32, 32)):
    """Helper to write synthetic GeoTIFF with explicit CRS and transform."""
    transform = from_origin(origin[0], origin[1], 10.0, 10.0)
    data = np.ones((4, shape[0], shape[1]), dtype=np.float32)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=shape[0], width=shape[1],
        count=4, dtype="float32",
        crs=crs, transform=transform
    ) as dst:
        dst.write(data)


def test_inverted_temporal_timestamps_raise_validation_error(temp_dir):
    """T1 timestamp (2024-01-01) occurs after T2 timestamp (2023-01-01)."""
    p1 = os.path.join(temp_dir, "t1.tif")
    p2 = os.path.join(temp_dir, "t2.tif")
    create_dummy_raster(p1)
    create_dummy_raster(p2)

    with pytest.raises(ValidationError) as exc:
        validate_input_pair(
            p1, p2,
            timestamp_t1="2024-01-01",
            timestamp_t2="2023-01-01"
        )
    assert "temporal_order" in str(exc.value).lower() or "precede" in str(exc.value).lower()



def test_non_overlapping_bounding_boxes_raise_validation_error(temp_dir):
    """Two rasters located 500 km apart must fail overlap validation."""
    p1 = os.path.join(temp_dir, "t1.tif")
    p2 = os.path.join(temp_dir, "t2.tif")
    create_dummy_raster(p1, origin=(500000, 1400000))
    create_dummy_raster(p2, origin=(900000, 2000000))  # Disjoint bounds

    with pytest.raises(ValidationError) as exc:
        validate_input_pair(p1, p2)
    assert "overlap" in str(exc.value).lower()


def test_missing_crs_raises_validation_error(temp_dir):
    """GeoTIFF without spatial reference system (CRS=None) must be trapped."""
    p = os.path.join(temp_dir, "nocrs.tif")
    create_dummy_raster(p, crs=None)

    info = check_raster_readable(p)
    with pytest.raises(ValidationError) as exc:
        check_crs_compatible(info, info)
    assert "crs" in str(exc.value).lower()


def test_corrupted_file_raises_validation_error(temp_dir):
    """Zero-byte or truncated file must fail validation cleanly."""
    corrupt_path = os.path.join(temp_dir, "corrupt.tif")
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_TIFF_FILE_HEADER")

    with pytest.raises(ValidationError) as exc:
        check_raster_readable(corrupt_path)
    assert "cannot open" in str(exc.value).lower() or "read error" in str(exc.value).lower()

