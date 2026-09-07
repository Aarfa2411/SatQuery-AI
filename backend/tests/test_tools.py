
import numpy as np
import pytest
from PIL import Image

from app.services.geo_compat import HAS_RASTERIO
from app.tools.tool1_vlm import vlm_tool
from app.tools.tool3_change import change_tool
from app.tools.tool4_fusion import fusion_tool


@pytest.fixture
def synthetic_pair(tmp_path):
    opt_path = tmp_path / "optical.tif"
    sar_path = tmp_path / "sar_risat.tif"

    opt_data = np.random.randint(50, 200, (80, 80, 3), dtype=np.uint8)
    sar_data = np.random.randint(10, 180, (80, 80), dtype=np.uint8)

    if HAS_RASTERIO:
        import rasterio
        from rasterio.transform import from_origin
        transform = from_origin(500000, 3000000, 1.0, 1.0)
        with rasterio.open(
            opt_path, 'w', driver='GTiff', height=80, width=80, count=3,
            dtype=np.uint8, crs='EPSG:32643', transform=transform
        ) as dst:
            dst.write(np.transpose(opt_data, (2, 0, 1)))

        with rasterio.open(
            sar_path, 'w', driver='GTiff', height=80, width=80, count=1,
            dtype=np.uint8, crs='EPSG:32643', transform=transform
        ) as dst:
            dst.write(sar_data, 1)
    else:
        Image.fromarray(opt_data).save(opt_path)
        Image.fromarray(sar_data).save(sar_path)

    return opt_path, sar_path

def test_tool1_taxonomy_mapping(synthetic_pair):
    opt_path, _ = synthetic_pair
    out = vlm_tool.execute([opt_path], {"query": "Where is the water reservoir?"})
    assert out.tool == "vqa_grounding"
    assert out.bboxes is not None
    assert len(out.bboxes) > 0
    assert "water" in out.answer.lower() or "reservoir" in out.answer.lower()
    assert 0.05 <= out.confidence <= 1.0

def test_tool3_change_with_pseudo_filter(synthetic_pair):
    opt_path, sar_path = synthetic_pair
    out = change_tool.execute([opt_path, sar_path], {"query": "What changed?", "suppress_pseudo_change": True})
    assert out.tool == "change_detection"
    assert out.mask_geojson is not None
    assert "pct_change" in out.metrics
    assert 0.05 <= out.confidence <= 1.0

def test_tool4_optical_sar_cloud_fallback(synthetic_pair):
    opt_path, sar_path = synthetic_pair
    out = fusion_tool.execute([opt_path, sar_path], {"query": "Assess ground inundation with radar"})
    assert out.tool == "optical_sar_fusion"
    assert "cloud_fraction" in out.metrics
    assert "sar_weight" in out.metrics
    assert 0.05 <= out.confidence <= 1.0
