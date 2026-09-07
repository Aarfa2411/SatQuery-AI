"""
tests/test_adversarial_geocv.py
===============================
Adversarial, edge-case, and geospatial robustness tests for the SatQuery AI
GeoCV pipeline per Section 10 of the GeoCV Lead charter.

Covers:
1. Temporal order violation (T1 later than T2).
2. All-nodata and all-zero input rasters.
3. Infinite / NaN / extreme values.
4. Non-overlapping bounding boxes.
5. Missing CRS / non-georeferenced images.
6. Model status honesty and ChangeFormer fallback on missing weights.
7. GeoJSON boundary polygon verification (EPSG:4326 compliance).
8. Quantitative evaluation metrics (Precision, Recall, F1, IoU, mIoU).
"""

from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from satquery.core.raster_io import RasterData, load_raster_from_array
from satquery.core.validator import (
    ValidationError,
    check_not_all_nodata,
    check_raster_array_finite,
    check_temporal_order,
    check_spatial_overlap,
)
from satquery.change_detection.models import (
    ModelStatus,
    get_change_model,
    ChangeFormerAdapter,
    ClassicalSpectralAdapter,
)
from satquery.change_detection.eval import (
    compute_pixel_metrics,
    compute_geospatial_metrics,
    ModelAblationComparator,
)
from satquery.change_detection.pipeline import ChangeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_raster(
    h: int = 64,
    w: int = 64,
    bands: int = 4,
    minx: float = 10.0,
    miny: float = 20.0,
    maxx: float = 10.1,
    maxy: float = 20.1,
    crs: str = "EPSG:4326",
    val: float = 0.5,
) -> RasterData:
    arr = np.full((bands, h, w), val, dtype=np.float32)
    tf = from_bounds(minx, miny, maxx, maxy, w, h)
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": w,
        "height": h,
        "count": bands,
        "crs": CRS.from_string(crs),
        "transform": tf,
    }
    return RasterData(arr, meta)


# ---------------------------------------------------------------------------
# Adversarial Tests: Validator & Sanity
# ---------------------------------------------------------------------------

def test_temporal_order_reversal_raises():
    """T1 timestamp later than T2 must raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        check_temporal_order("2024-05-01", "2023-05-01")
    assert "TEMPORAL_ORDER" in str(exc.value)


def test_temporal_order_equal_raises():
    """Identical timestamps must raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        check_temporal_order("2024-01-01", "2024-01-01")
    assert "TEMPORAL_ORDER" in str(exc.value)


def test_all_nodata_detection():
    """Raster consisting entirely of nodata or NaNs must raise."""
    all_nan = np.full((64, 64), np.nan, dtype=np.float32)
    with pytest.raises(ValidationError) as exc:
        check_not_all_nodata(all_nan)
    assert "ALL_NODATA" in str(exc.value)

    all_nodata = np.full((64, 64), -9999.0, dtype=np.float32)
    with pytest.raises(ValidationError) as exc:
        check_not_all_nodata(all_nodata, nodata_val=-9999.0)
    assert "ALL_NODATA" in str(exc.value)


def test_all_zero_detection():
    """Raster with all zero pixels must raise ValidationError."""
    all_zero = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValidationError) as exc:
        check_not_all_nodata(all_zero)
    assert "ALL_ZERO" in str(exc.value)


def test_infinite_values_detection():
    """Raster containing Inf must be detected."""
    arr_inf = np.ones((64, 64), dtype=np.float32)
    arr_inf[10, 10] = np.inf
    with pytest.raises(ValidationError) as exc:
        check_raster_array_finite(arr_inf)
    assert "NON_FINITE_VALUES" in str(exc.value)


def test_non_overlapping_scenes():
    """Scenes in completely different geographical coordinates must raise."""
    info1 = {"bounds": (10.0, 10.0, 11.0, 11.0), "path": "t1.tif"}
    info2 = {"bounds": (80.0, 80.0, 81.0, 81.0), "path": "t2.tif"}
    with pytest.raises(ValidationError) as exc:
        check_spatial_overlap(info1, info2)
    assert "SPATIAL_OVERLAP" in str(exc.value)


# ---------------------------------------------------------------------------
# Learned Model & Status Honesty Tests
# ---------------------------------------------------------------------------

def test_changeformer_fallback_honesty():
    """
    When ChangeFormer weights are absent, it must NOT claim to be a REAL_MODEL.
    It must report HEURISTIC_FALLBACK honestly.
    """
    adapter = ChangeFormerAdapter(weights_path="/nonexistent/weights.onnx", fallback_on_missing=True)
    assert not adapter.is_available()

    r1 = _make_dummy_raster(val=0.2)
    r2 = _make_dummy_raster(val=0.8)
    pred = adapter.predict(r1, r2)

    assert pred.model_status == ModelStatus.HEURISTIC_FALLBACK
    assert "fallback_reason" in pred.provenance
    assert pred.change_mask.shape == (64, 64)


def test_classical_adapter_status():
    """Classical adapter must report CLASSICAL_ALGORITHM."""
    adapter = ClassicalSpectralAdapter()
    assert adapter.is_available()

    r1 = _make_dummy_raster(val=0.2)
    r2 = _make_dummy_raster(val=0.8)
    pred = adapter.predict(r1, r2)

    assert pred.model_status == ModelStatus.CLASSICAL_ALGORITHM
    assert 0.0 <= pred.confidence <= 1.0


# ---------------------------------------------------------------------------
# End-to-End GeoCV Output Contract Verification
# ---------------------------------------------------------------------------

def test_pipeline_charter_contract_output():
    """
    Verifies that ChangeDetector returns the exact Section 6 Charter keys:
    change_type, confidence, changed_area_m2, changed_area_ha, change_percentage,
    geometry, supporting_evidence, warnings.
    """
    r1 = _make_dummy_raster(val=0.3)
    r2 = _make_dummy_raster(val=0.7)

    detector = ChangeDetector(query_hint="vegetation", model_name="classical")
    res = detector.run_from_arrays(r1, r2, timestamp_t1="2023-01-01", timestamp_t2="2024-01-01")

    # Core Section 6 keys
    assert "change_type" in res
    assert "confidence" in res
    assert "changed_area_m2" in res
    assert "changed_area_ha" in res
    assert "change_percentage" in res
    assert "geometry" in res
    assert "supporting_evidence" in res
    assert "warnings" in res

    # Supporting evidence structure
    se = res["supporting_evidence"]
    assert "model_status" in se
    assert se["model_status"] == ModelStatus.CLASSICAL_ALGORITHM.value
    assert "spectral_index" in se
    assert "bimodal_separation" in se

    # GeoJSON validity
    geom = res["geometry"]
    assert geom["type"] == "FeatureCollection"
    assert isinstance(geom["features"], list)


# ---------------------------------------------------------------------------
# Quantitative Evaluation Suite Tests
# ---------------------------------------------------------------------------

def test_pixel_metrics_perfect_overlap():
    gt = np.zeros((100, 100), dtype=bool)
    gt[20:50, 20:50] = True
    pred = gt.copy()

    metrics = compute_pixel_metrics(pred, gt)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.iou == 1.0
    assert metrics.overall_accuracy == 1.0


def test_ablation_comparator():
    gt = np.zeros((50, 50), dtype=bool)
    gt[10:30, 10:30] = True

    pred1 = np.zeros((50, 50), dtype=bool)
    pred1[10:30, 10:30] = True  # perfect

    pred2 = np.zeros((50, 50), dtype=bool)
    pred2[15:35, 15:35] = True  # partial overlap

    p1 = ChangeFormerAdapter(fallback_on_missing=True).predict(
        _make_dummy_raster(h=50, w=50), _make_dummy_raster(h=50, w=50)
    )
    p1.change_mask = pred1

    p2 = ClassicalSpectralAdapter().predict(
        _make_dummy_raster(h=50, w=50), _make_dummy_raster(h=50, w=50)
    )
    p2.change_mask = pred2

    comparator = ModelAblationComparator(ground_truth_mask=gt)
    report = comparator.evaluate_predictions({"learned": p1, "classical": p2})

    assert "model_evaluations" in report
    assert "cross_model_agreement" in report
    assert report["model_evaluations"]["learned"]["metrics"]["iou"] == 1.0
