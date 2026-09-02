"""
tests/test_change_pipeline.py
==============================
Unit and integration tests for the SatQuery AI change detection pipeline.

Test strategy
-------------
All tests use synthetic in-memory numpy arrays (no real GeoTIFF files needed)
so they run instantly without I/O and are safe for CI environments.

The synthetic scenarios mirror the three demo scenarios:
  - Flood     : NDWI increases in a rectangular region
  - Deforest  : NDVI decreases in a rectangular region
  - Urban     : NDBI increases in a rectangular region
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
from satquery.core.raster_io import load_raster_from_array
from satquery.change_detection.indices import (
    ndvi, ndwi, ndbi, rvi,
    extract_index, available_indices,
    sar_amplitude_to_db,
)
from satquery.change_detection.detector import (
    compute_difference,
    smooth_difference,
    suppress_pseudo_changes,
    otsu_threshold,
    detect_changes,
)
from satquery.change_detection.morphology import clean_mask, label_components
from satquery.change_detection.confidence import compute_confidence, confidence_label
from satquery.change_detection.metrics import (
    compute_area, classify_change_direction, build_text_summary
)
from satquery.change_detection.pipeline import ChangeDetector
from satquery.core.validator import ValidationError, check_temporal_order


# ============================================================
#  Fixtures: synthetic rasters (H=64, W=64, 5-band optical)
# ============================================================

H, W = 64, 64  # small test images

def _make_optical_raster(ndvi_value: float = 0.4, ndwi_value: float = -0.2) -> np.ndarray:
    """
    Create a (5, H, W) float32 array simulating an optical raster.
    Band layout: [Blue, Green, Red, NIR, SWIR]
    Derives band values from target NDVI and NDWI.
    """
    arr = np.zeros((5, H, W), dtype=np.float32)
    # NIR and Red from NDVI target
    nir  = 0.6
    red  = nir * (1 - ndvi_value) / (1 + ndvi_value)
    # Green from NDWI target
    green = (ndwi_value + 1) * nir / (1 - ndwi_value)
    arr[0] = 0.05                   # Blue
    arr[1] = float(green)           # Green
    arr[2] = float(red)             # Red
    arr[3] = float(nir)             # NIR
    arr[4] = 0.2                    # SWIR
    return arr


def _make_flood_scenario() -> tuple[np.ndarray, np.ndarray]:
    """
    T1: mostly vegetated (NDWI=-0.3)
    T2: flood in central 20×20 patch (NDWI=+0.5 in that region)
    """
    t1 = _make_optical_raster(ndvi_value=0.5, ndwi_value=-0.3)
    t2 = t1.copy()
    # Increase NDWI in flood patch by reducing Red/NIR, increasing Green
    patch = (slice(22, 42), slice(22, 42))
    t2[3][patch] = 0.3   # lower NIR
    t2[1][patch] = 0.5   # higher Green → NDWI up
    return t1, t2


def _make_deforestation_scenario() -> tuple[np.ndarray, np.ndarray]:
    """
    T1: high NDVI (0.7)
    T2: NDVI drops to 0.1 in top-left 24×24 patch (clear-cut)
    """
    t1 = _make_optical_raster(ndvi_value=0.7, ndwi_value=-0.2)
    t2 = t1.copy()
    patch = (slice(0, 24), slice(0, 24))
    t2[2][patch] = 0.55   # high Red (bare soil)
    t2[3][patch] = 0.15   # low NIR
    return t1, t2


def _make_no_change_scenario() -> tuple[np.ndarray, np.ndarray]:
    """Identical T1 and T2 (no change at all)."""
    t1 = _make_optical_raster(ndvi_value=0.4, ndwi_value=-0.1)
    t2 = t1.copy()
    return t1, t2


# ============================================================
#  1. Spectral index tests
# ============================================================

class TestSpectralIndices:
    def test_ndvi_range(self):
        red = np.array([[0.2, 0.5, 0.1]])
        nir = np.array([[0.6, 0.4, 0.8]])
        result = ndvi(red, nir)
        assert result.shape == (1, 3)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_ndvi_known_value(self):
        red = np.array([[0.2]])
        nir = np.array([[0.8]])
        # (0.8 - 0.2) / (0.8 + 0.2) = 0.6
        result = ndvi(red, nir)
        assert abs(result[0, 0] - 0.6) < 1e-5

    def test_ndwi_range(self):
        green = np.random.rand(10, 10).astype(np.float32)
        nir   = np.random.rand(10, 10).astype(np.float32)
        result = ndwi(green, nir)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_ndbi_zero_division(self):
        """Degenerate case: SWIR = NIR → NDBI = 0."""
        swir = nir = np.ones((5, 5), dtype=np.float32) * 0.5
        result = ndbi(swir, nir)
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_rvi_range(self):
        """RVI = 4·VH/(VV+VH): always ≥ 0; can exceed 1 for raw amplitude values."""
        vv = np.abs(np.random.rand(8, 8).astype(np.float32)) + 0.01
        vh = np.abs(np.random.rand(8, 8).astype(np.float32)) + 0.01
        result = rvi(vv, vh)
        assert np.all(result >= 0.0), "RVI must be non-negative"

    def test_sar_amplitude_to_db(self):
        """10·log10(1.0) = 0 dB."""
        amp = np.ones((4, 4), dtype=np.float32)
        result = sar_amplitude_to_db(amp)
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_extract_index_ndvi(self):
        arr = _make_optical_raster(ndvi_value=0.5)
        result = extract_index(arr, "ndvi")
        assert result is not None
        assert result.shape == (H, W)
        assert np.allclose(result.mean(), 0.5, atol=0.02)

    def test_extract_index_unavailable(self):
        """Single-band array should return None for NDVI."""
        arr = np.random.rand(1, 20, 20).astype(np.float32)
        result = extract_index(arr, "ndvi")
        assert result is None

    def test_available_indices_optical_5band(self):
        arr = np.zeros((5, 10, 10), dtype=np.float32)
        indices = available_indices(arr, sensor="optical")
        assert "ndvi" in indices
        assert "ndwi" in indices
        assert "ndbi" in indices

    def test_available_indices_sar_2band(self):
        arr = np.zeros((2, 10, 10), dtype=np.float32)
        indices = available_indices(arr, sensor="sar")
        assert "rvi" in indices


# ============================================================
#  2. Detector tests
# ============================================================

class TestDetector:
    def test_compute_difference_absolute(self):
        a = np.ones((10, 10), dtype=np.float32) * 0.3
        b = np.ones((10, 10), dtype=np.float32) * 0.7
        diff = compute_difference(a, b, mode="absolute")
        assert np.allclose(diff, 0.4, atol=1e-5)
        assert np.all(diff >= 0)

    def test_compute_difference_signed(self):
        a = np.ones((5, 5), dtype=np.float32) * 0.3
        b = np.ones((5, 5), dtype=np.float32) * 0.1
        diff = compute_difference(a, b, mode="signed")
        assert np.allclose(diff, -0.2, atol=1e-5)

    def test_compute_difference_shape_mismatch(self):
        a = np.ones((10, 10), dtype=np.float32)
        b = np.ones((8, 12), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_difference(a, b)

    def test_smooth_difference_reduces_range(self):
        diff = np.random.rand(32, 32).astype(np.float32)
        diff[16, 16] = 100.0  # spike
        smoothed = smooth_difference(diff, sigma=2.0)
        # Spike should be reduced
        assert smoothed[16, 16] < diff[16, 16]

    def test_smooth_difference_zero_sigma(self):
        diff = np.random.rand(10, 10).astype(np.float32)
        smoothed = smooth_difference(diff, sigma=0)
        assert np.allclose(smoothed, diff)

    def test_otsu_threshold_uniform(self):
        """Uniform diff map — should warn and return empty mask."""
        diff = np.ones((20, 20), dtype=np.float32) * 0.5
        mask, thresh = otsu_threshold(diff)
        assert mask.sum() == 0 or thresh == 0.0

    def test_otsu_threshold_bimodal(self):
        """Clearly bimodal diff map — should produce a clean mask."""
        diff = np.zeros((50, 50), dtype=np.float32)
        diff[10:30, 10:30] = 0.8   # change region
        mask, thresh = otsu_threshold(diff)
        assert mask.sum() > 0
        assert 0.0 < thresh < 0.8

    def test_suppress_pseudo_changes_uniform_shift(self):
        """
        Simulate a pure radiometric shift (same change everywhere, flat neighbourhoods).
        Pseudo-change filter should zero out most of the difference.
        """
        idx_t1 = np.ones((40, 40), dtype=np.float32) * 0.4
        idx_t2 = np.ones((40, 40), dtype=np.float32) * 0.7   # uniform +0.3 shift
        diff   = np.abs(idx_t2 - idx_t1)
        suppressed = suppress_pseudo_changes(diff, idx_t1, idx_t2, window_size=7)
        # Most pixels should be zeroed as pseudo-change
        frac_zeroed = (suppressed == 0).mean()
        assert frac_zeroed > 0.5, f"Expected >50% suppressed, got {frac_zeroed:.2%}"

    def test_detect_changes_no_change(self):
        t1_arr, t2_arr = _make_no_change_scenario()
        idx_t1 = extract_index(t1_arr, "ndvi")
        idx_t2 = extract_index(t2_arr, "ndvi")
        result = detect_changes(idx_t1, idx_t2, smooth_sigma=0, suppress_pseudo_changes=False)
        assert result["change_mask"].sum() == 0


# ============================================================
#  3. Morphology tests
# ============================================================

class TestMorphology:
    def test_clean_mask_removes_single_pixel(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[15, 15] = True   # single isolated pixel
        cleaned = clean_mask(mask, open_radius=2, close_radius=0, min_area_px=4)
        assert not cleaned[15, 15], "Single pixel should be removed by opening"

    def test_label_components_count(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[5:10, 5:10] = True    # region 1
        mask[30:40, 30:40] = True  # region 2
        labeled, n = label_components(mask)
        assert n == 2

    def test_clean_mask_preserves_large_region(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True  # large 30×30 region
        cleaned = clean_mask(mask, open_radius=2, min_area_px=4)
        assert cleaned[25, 25], "Large region should survive cleaning"


# ============================================================
#  4. Confidence scoring tests
# ============================================================

class TestConfidence:
    def test_confidence_uniform_is_zero(self):
        diff = np.ones((30, 30), dtype=np.float32) * 0.5
        mask = np.zeros((30, 30), dtype=bool)
        score = compute_confidence(diff, mask, otsu_threshold=0.5)
        assert score == 0.0

    def test_confidence_bimodal_is_high(self):
        diff = np.zeros((60, 60), dtype=np.float32)
        diff[20:40, 20:40] = 0.9   # clear bimodal split
        mask = diff > 0.5
        score = compute_confidence(diff, mask, otsu_threshold=0.45)
        assert score > 0.5, f"Expected high confidence, got {score}"

    def test_confidence_label_thresholds(self):
        assert confidence_label(0.9)  == "HIGH"
        assert confidence_label(0.6)  == "MEDIUM"
        assert confidence_label(0.3)  == "LOW"
        assert confidence_label(0.1)  == "VERY_LOW"


# ============================================================
#  5. Metrics tests
# ============================================================

class TestMetrics:
    def _dummy_transform(self, gsd: float = 10.0):
        """Create a minimal Affine transform with given GSD (metres)."""
        import affine
        return affine.Affine(gsd, 0.0, 0.0, 0.0, -gsd, gsd * 100)

    def test_compute_area_known(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True   # 4×4 = 16 pixels
        transform = self._dummy_transform(gsd=10.0)   # 10m pixels
        area = compute_area(mask, transform)
        assert area["n_changed_pixels"] == 16
        assert abs(area["area_m2"] - 16 * 100.0) < 1.0   # 16 × 100 m²

    def test_classify_no_change(self):
        diff = np.zeros((10, 10), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=bool)
        label = classify_change_direction(diff, mask, primary_index="ndvi")
        assert label == "no_change"

    def test_classify_vegetation_loss(self):
        diff = np.full((20, 20), -0.4, dtype=np.float32)  # NDVI dropped
        mask = np.ones((20, 20), dtype=bool)
        label = classify_change_direction(diff, mask, primary_index="ndvi")
        assert label == "vegetation_loss"

    def test_classify_water_expansion(self):
        diff = np.full((20, 20), 0.35, dtype=np.float32)  # NDWI increased
        mask = np.ones((20, 20), dtype=bool)
        label = classify_change_direction(diff, mask, primary_index="ndwi")
        assert label == "water_expansion"

    def test_build_text_summary(self):
        area_metrics = {
            "area_ha": 12.5, "area_m2": 125000,
            "n_changed_pixels": 500, "pct_changed": 3.5,
        }
        text = build_text_summary(
            area_metrics, "vegetation_loss", "ndvi",
            n_regions=3, timestamp_t1="2023-01-01", timestamp_t2="2024-01-01",
            confidence=0.82,
        )
        assert "vegetation loss" in text.lower()
        assert "12.50 ha" in text
        assert "0.82" in text


# ============================================================
#  6. GeoValidator tests
# ============================================================

class TestValidator:
    def test_temporal_order_valid(self):
        check_temporal_order("2023-01-01", "2024-06-15")  # Should not raise

    def test_temporal_order_inverted(self):
        with pytest.raises(ValidationError) as exc_info:
            check_temporal_order("2024-06-15", "2023-01-01")
        assert "TEMPORAL_ORDER" in str(exc_info.value)

    def test_temporal_order_same_date(self):
        with pytest.raises(ValidationError):
            check_temporal_order("2023-06-01", "2023-06-01")

    def test_temporal_order_none_passes(self):
        check_temporal_order(None, "2024-01-01")  # Should not raise


# ============================================================
#  7. End-to-end pipeline tests (in-memory)
# ============================================================

class TestPipelineEndToEnd:
    def _run(self, t1_arr, t2_arr, query_hint="", sensor="optical"):
        raster_t1 = load_raster_from_array(t1_arr)
        raster_t2 = load_raster_from_array(t2_arr)
        detector = ChangeDetector(
            sensor_t1=sensor, sensor_t2=sensor,
            query_hint=query_hint,
            suppress_pseudo=True,
        )
        return detector.run_from_arrays(raster_t1, raster_t2)

    def test_flood_scenario(self):
        t1, t2 = _make_flood_scenario()
        result = self._run(t1, t2, query_hint="flood")
        assert result["status"] in ("ok", "partial")
        assert result["primary_index"] == "ndwi"
        assert result["change_direction"] in ("water_expansion", "mixed_change", "no_change")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "summary" in result
        assert result["geojson"]["type"] == "FeatureCollection"
        assert len(result["execution_trace"]) >= 5

    def test_deforestation_scenario(self):
        t1, t2 = _make_deforestation_scenario()
        result = self._run(t1, t2, query_hint="forest")
        assert result["status"] in ("ok", "partial")
        assert result["primary_index"] == "ndvi"
        assert result["change_direction"] in ("vegetation_loss", "mixed_change", "no_change")

    def test_no_change_scenario(self):
        t1, t2 = _make_no_change_scenario()
        result = self._run(t1, t2)
        assert result["status"] in ("ok", "partial")
        # Changed pixels may be zero or very small
        assert result["n_changed_pixels"] < (H * W * 0.05)

    def test_execution_trace_has_why_annotations(self):
        t1, t2 = _make_flood_scenario()
        result = self._run(t1, t2, query_hint="flood")
        trace = result["execution_trace"]
        why_annotated = [step for step in trace if "why" in step]
        assert len(why_annotated) >= 2, "Expected at least 2 trace steps with 'why' annotations"

    def test_result_schema_completeness(self):
        """All required keys must be present in the result dict."""
        t1, t2 = _make_flood_scenario()
        result = self._run(t1, t2)
        required_keys = [
            "status", "primary_index", "change_direction",
            "confidence", "confidence_label", "area_metrics",
            "n_changed_pixels", "n_regions", "otsu_threshold",
            "n_pseudo_removed", "summary", "geojson",
            "execution_trace", "warnings", "sensor_calibration_note",
            "total_processing_ms",
        ]
        missing = [k for k in required_keys if k not in result]
        assert not missing, f"Missing result keys: {missing}"

    def test_sensor_note_cartosat(self):
        t1, t2 = _make_flood_scenario()
        raster_t1 = load_raster_from_array(t1)
        raster_t2 = load_raster_from_array(t2)
        detector = ChangeDetector(sensor_t1="cartosat", sensor_t2="cartosat")
        result = detector.run_from_arrays(raster_t1, raster_t2)
        assert "Cartosat" in result["sensor_calibration_note"]
