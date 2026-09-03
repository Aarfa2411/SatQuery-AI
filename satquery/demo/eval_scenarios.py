"""
satquery.demo.eval_scenarios
=============================
Evaluates the ChangeDetector pipeline against all generated demo scenarios.

Runs the pipeline on the GeoTIFF files in demo_data/ and prints an
executive benchmark summary table with metrics, confidence, and validation
against expected scenario metadata.
"""

import json
import os
import time
from satquery.change_detection.pipeline import ChangeDetector

def run_evaluation(demo_dir: str = "./demo_data") -> list[dict]:
    index_path = os.path.join(demo_dir, "scenarios_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Scenario index not found at {index_path}. Run generate_scenarios first.")

    with open(index_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    results = []
    print("\n" + "=" * 80)
    print(" SatQuery AI — Bi-Temporal Change Detection Evaluation ")
    print("=" * 80)

    for item in scenarios:
        s_name = item["scenario"]
        t1_path = item["t1_path"]
        t2_path = item["t2_path"]
        meta_path = item["meta_path"]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        query_hint = meta.get("query_hint", "")
        t1_date = meta.get("timestamp_t1")
        t2_date = meta.get("timestamp_t2")

        detector = ChangeDetector(
            sensor_t1="cartosat",
            sensor_t2="cartosat",
            query_hint=query_hint,
            suppress_pseudo=True,
        )

        t0 = time.perf_counter()
        out = detector.run_from_files(
            t1_path, t2_path,
            timestamp_t1=t1_date,
            timestamp_t2=t2_date,
            output_dir=os.path.join(demo_dir, s_name, "output"),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        expected = meta.get("expected", {})
        idx_match = (out["primary_index"] == expected.get("primary_index"))
        dir_match = (out["change_direction"] == expected.get("change_direction"))

        res_summary = {
            "scenario": s_name,
            "title": meta.get("title"),
            "status": out["status"],
            "primary_index": out["primary_index"],
            "expected_index": expected.get("primary_index"),
            "index_match": idx_match,
            "direction": out["change_direction"],
            "expected_direction": expected.get("change_direction"),
            "direction_match": dir_match,
            "confidence": out["confidence"],
            "confidence_label": out["confidence_label"],
            "area_ha": out["area_metrics"]["area_ha"],
            "n_regions": out["n_regions"],
            "n_pseudo_removed": out["n_pseudo_removed"],
            "processing_ms": round(elapsed_ms, 1),
            "geojson_features": len(out["geojson"]["features"]),
        }
        results.append(res_summary)

        print(f"\nScenario: [{s_name.upper()}] — {meta.get('title')}")
        print(f"  - Status            : {out['status']}")
        print(f"  - Primary Index     : {out['primary_index'].upper()} (Expected: {expected.get('primary_index', '').upper()}) -> Match: {'YES' if idx_match else 'NO'}")
        print(f"  - Direction         : {out['change_direction']} (Expected: {expected.get('change_direction')}) -> Match: {'YES' if dir_match else 'NO'}")
        print(f"  - Confidence Score  : {out['confidence']:.4f} ({out['confidence_label']})")
        print(f"  - Changed Area      : {out['area_metrics']['area_ha']:.3f} ha ({out['area_metrics']['pct_changed']:.2f}% of area)")
        print(f"  - GeoJSON Features  : {len(out['geojson']['features'])} polygons")
        print(f"  - Pseudo-Suppressed : {out['n_pseudo_removed']} pixels")
        print(f"  - Latency           : {elapsed_ms:.1f} ms")
        print(f"  - Summary Text      : {out['summary']}")

    print("\n" + "=" * 80)
    print(" Evaluation Completed Successfully ")
    print("=" * 80 + "\n")
    return results

if __name__ == "__main__":
    run_evaluation()
