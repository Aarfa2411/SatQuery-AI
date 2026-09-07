"""
Fast Pre-Merge End-to-End Smoke Test (<15 Seconds)
===================================================
NOTE: This test suite uses small 256x256 crop fixtures specifically chosen to make
the <15s target achievable as a fast CI pre-merge gate. For full production-scale
latency benchmarks, see `satquery.evaluation.harness`.
"""

import time
import numpy as np
import rasterio
from rasterio.transform import from_origin
from satquery.core.raster_io import RasterData
from satquery.change_detection.pipeline import ChangeDetector
from satquery.evaluation.adapters import ChangeDetectionAdapter
from satquery.evaluation.guardrails import NumericalGuardrail


def test_fast_e2e_change_detection_smoke():
    """
    Executes 256x256 synthetic crop pair through:
    ChangeDetector -> Adapter -> Guardrail in < 15 seconds.
    """
    t_start = time.perf_counter()

    # 1. Generate in-memory 256x256 4-band rasters (Blue, Green, Red, NIR)
    shape = (256, 256)
    transform = from_origin(500000, 1400000, 10.0, 10.0)
    crs = rasterio.crs.CRS.from_epsg(32643)

    # T1: Baseline vegetation (NIR high, Red low)
    t1_data = np.ones((4, shape[0], shape[1]), dtype=np.float32) * 500.0
    t1_data[2, :, :] = 400.0   # Red
    t1_data[3, :, :] = 2500.0  # NIR (high NDVI)

    # T2: Deforestation in central 64x64 patch (NIR drops)
    t2_data = t1_data.copy()
    t2_data[3, 96:160, 96:160] = 600.0  # NIR drops significantly

    meta = {
        "crs": crs,
        "transform": transform,
        "count": 4,
        "height": shape[0],
        "width": shape[1],
        "dtype": "float32",
    }
    r1 = RasterData(t1_data, meta=meta)
    r2 = RasterData(t2_data, meta=meta)

    # 2. Run ChangeDetector
    detector = ChangeDetector(query_hint="deforestation", smooth_sigma=1.0)
    raw_result = detector.run_from_arrays(r1, r2, timestamp_t1="2023-01-01", timestamp_t2="2024-01-01")



    assert raw_result["status"] == "ok"
    area_val = raw_result["area_metrics"].get("area_ha", raw_result["area_metrics"].get("changed_area_ha", 0.0))
    assert area_val > 0.0


    # 3. Adapt to EvaluationRecord schema
    record = ChangeDetectionAdapter.parse(raw_result, task_id="SMOKE-TEST-001")
    assert record.status == "classical_algorithm"
    assert record.confidence_score > 0.0
    assert record.confidence_breakdown is not None

    # 4. Guardrail verification on summary text
    guardrail = NumericalGuardrail(tolerance=0.05)
    passed, violations, _ = guardrail.verify(
        record.text_response,
        record.deterministic_numbers,
        geojson_geometry=record.geojson_geometry
    )
    assert passed is True

    elapsed = time.perf_counter() - t_start
    print(f"\n[Smoke Test] Completed in {elapsed:.2f}s (<15.0s target).")
    assert elapsed < 15.0
