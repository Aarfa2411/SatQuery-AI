
import numpy as np
import pytest
from PIL import Image

from app.services.geo_compat import HAS_RASTERIO
from app.services.input_validator import input_validator


@pytest.fixture
def sample_geotiffs(tmp_path):
    """Generates two test GeoTIFFs with spatial metadata."""
    t1_path = tmp_path / "cartosat_t1_20240301.tif"
    t2_path = tmp_path / "cartosat_t2_20240901.tif"

    data1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    data2 = np.ones((100, 100, 3), dtype=np.uint8) * 150

    if HAS_RASTERIO:
        import rasterio
        from rasterio.transform import from_origin
        transform = from_origin(500000, 3000000, 2.0, 2.0)
        for path, data in [(t1_path, np.transpose(data1, (2, 0, 1))), (t2_path, np.transpose(data2, (2, 0, 1)))]:
            with rasterio.open(
                path, 'w', driver='GTiff', height=100, width=100, count=3,
                dtype=data.dtype, crs='EPSG:32643', transform=transform
            ) as dst:
                dst.write(data)
    else:
        Image.fromarray(data1).save(t1_path)
        Image.fromarray(data2).save(t2_path)

    return t1_path, t2_path

def test_single_image_inspection(sample_geotiffs):
    t1_path, _ = sample_geotiffs
    meta, warnings, errors = input_validator.inspect_file(t1_path, "test_1")
    assert errors == []
    assert meta is not None
    assert meta.crs == "EPSG:32643"
    assert meta.resolution_m > 0
    assert meta.bands == 3
    assert meta.timestamp == "2024-03-01"

def test_bi_temporal_validation_and_coregistration(sample_geotiffs):
    t1_path, t2_path = sample_geotiffs
    result = input_validator.validate_session_inputs("test_session", [t1_path, t2_path])
    assert result.valid is True
    assert result.mode == "bi_temporal"
    assert result.is_coregistered is True
    assert result.overlap_pct == 100.0

def test_reject_corrupt_file(tmp_path):
    corrupt_path = tmp_path / "corrupt.tif"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_TIFF_FILE_HEADER")
    
    meta, warnings, errors = input_validator.inspect_file(corrupt_path, "corrupt_id")
    assert len(errors) > 0
    assert meta is None
