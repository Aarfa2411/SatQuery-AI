"""
satquery.demo.generate_scenarios
==================================
Synthetic GeoTIFF scenario generator for the three primary judging demos.

Scenarios
---------
1. Flood Inundation       — NDWI-driven water expansion in a river floodplain
2. Deforestation          — NDVI-driven vegetation clearance (selective logging pattern)
3. Urban Expansion        — NDBI-driven new settlement growth on agricultural land

Each scenario produces:
  - T1 and T2 GeoTIFF files (5-band optical, 256×256 pixels)
  - A ``scenario_meta.json`` describing expected outputs for evaluation

Usage
-----
    python -m satquery.demo.generate_scenarios --output-dir ./demo_data

Or programmatically:
    from satquery.demo.generate_scenarios import generate_all_scenarios
    generate_all_scenarios(output_dir="./demo_data")
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H, W   = 256, 256          # image dimensions (pixels)
GSD_M  = 2.0               # ground sample distance in metres (Cartosat-2S MSS)
ORIGIN = (80.0, 18.0)      # top-left corner (lon, lat) — Bay of Bengal coast, India

# Affine transform: 2 m pixels, UTM-like (using geographic CRS for simplicity)
_DEG_PER_M = 1 / 111_319.0   # approximate degrees per metre at equator
GSD_DEG    = GSD_M * _DEG_PER_M

BASE_TRANSFORM = Affine(GSD_DEG, 0.0, ORIGIN[0],
                        0.0, -GSD_DEG, ORIGIN[1])
BASE_CRS       = CRS.from_epsg(4326)


# ---------------------------------------------------------------------------
# Band naming convention (0-indexed)
# ---------------------------------------------------------------------------
# Band 0: Blue      (~450 nm)
# Band 1: Green     (~560 nm)
# Band 2: Red       (~665 nm)
# Band 3: NIR       (~865 nm)
# Band 4: SWIR      (~1610 nm)
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR"]


# ---------------------------------------------------------------------------
# Reflectance synthesis helpers
# ---------------------------------------------------------------------------

def _noise(shape: tuple, scale: float = 0.01) -> np.ndarray:
    """Gaussian noise to add realism (sensor noise)."""
    return (np.random.randn(*shape) * scale).astype(np.float32)


def _vegetation_pixel(ndvi: float = 0.65) -> np.ndarray:
    """Return approximate band reflectances for a vegetated pixel."""
    nir  = 0.55
    red  = nir * (1 - ndvi) / (1 + ndvi)
    return np.array([0.04, 0.07, float(red), float(nir), 0.12], dtype=np.float32)


def _water_pixel() -> np.ndarray:
    """Return approximate band reflectances for an open water pixel."""
    return np.array([0.06, 0.05, 0.03, 0.02, 0.01], dtype=np.float32)


def _bare_soil_pixel() -> np.ndarray:
    """Return approximate band reflectances for exposed bare soil."""
    return np.array([0.18, 0.20, 0.22, 0.25, 0.28], dtype=np.float32)


def _urban_pixel() -> np.ndarray:
    """Return approximate band reflectances for built-up surfaces."""
    return np.array([0.20, 0.22, 0.24, 0.22, 0.35], dtype=np.float32)


def _fill_background(arr: np.ndarray, land_type: str = "vegetation") -> None:
    """Fill the entire image with a background land-cover type."""
    px_map = {
        "vegetation": _vegetation_pixel(),
        "water":      _water_pixel(),
        "bare_soil":  _bare_soil_pixel(),
        "urban":      _urban_pixel(),
    }
    px = px_map.get(land_type, _vegetation_pixel())
    for b in range(5):
        arr[b, :, :] = px[b]


def _apply_patch(arr: np.ndarray, row_sl: slice, col_sl: slice, land_type: str) -> None:
    """Paint a rectangular patch with a given land-cover type."""
    px_map = {
        "vegetation": _vegetation_pixel(),
        "water":      _water_pixel(),
        "bare_soil":  _bare_soil_pixel(),
        "urban":      _urban_pixel(),
    }
    px = px_map[land_type]
    for b in range(5):
        arr[b, row_sl, col_sl] = px[b]


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def _make_flood_scenario() -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Flood Inundation Scenario
    -------------------------
    Context : Coastal river-delta area after a cyclone.
    T1       : Mixed vegetation + existing river channel (narrow).
    T2       : River expands massively; agricultural land inundated.

    Expected output:
        change_direction = "water_expansion"
        primary_index    = "ndwi"
        area             ≈ 88 ha (at 2 m GSD, 64×64 px patch = 4096 px × 4 m² = 16,384 m²)
    """
    t1 = np.zeros((5, H, W), dtype=np.float32)
    t2 = np.zeros((5, H, W), dtype=np.float32)

    # Background: green agricultural land
    _fill_background(t1, "vegetation")
    _fill_background(t2, "vegetation")

    # T1: narrow river channel (20 px wide, central vertical strip)
    _apply_patch(t1, slice(0, H), slice(120, 140), "water")

    # T2: flood expands river to 120 px width + inundates left fields
    _apply_patch(t2, slice(0, H), slice(60, 180), "water")

    # Additional flood pool in top-right quadrant (storm surge)
    _apply_patch(t2, slice(10, 80), slice(180, 240), "water")

    # Add sensor noise
    t1 += _noise((5, H, W), scale=0.008)
    t2 += _noise((5, H, W), scale=0.008)
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)

    # Changed area: new water pixels in T2 that were not water in T1
    new_water_px = (120 - 60) * H + (240 - 180) * 70   # approximate
    area_m2 = new_water_px * (GSD_M ** 2)

    meta: dict[str, Any] = {
        "scenario":          "flood_inundation",
        "title":             "Cyclone Flood Inundation — River Delta",
        "location":          "Bay of Bengal Coast, Odisha (synthetic)",
        "timestamp_t1":      "2023-10-10",
        "timestamp_t2":      "2023-10-12",
        "sensor":            "optical",
        "sensor_label":      "Cartosat-2S MSS (synthetic)",
        "query_hint":        "flood",
        "expected": {
            "primary_index":    "ndwi",
            "change_direction": "water_expansion",
            "approx_area_m2":   area_m2,
        },
        "description": (
            "Simulates the 2023 cyclone-driven coastal flooding scenario. "
            "The river channel expands from 40 m to 240 m width, inundating "
            "surrounding paddy fields and coastal settlements. "
            "Query this scenario with: 'Has there been flooding between T1 and T2?'"
        ),
    }
    return t1, t2, meta


def _make_deforestation_scenario() -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Deforestation Scenario
    ----------------------
    Context : Western Ghats forest edge — selective logging and clearance.
    T1       : Dense forested hillside (high NDVI ≈ 0.75).
    T2       : Logging tracks, cleared patches, exposed soil visible.

    Expected output:
        change_direction = "vegetation_loss"
        primary_index    = "ndvi"
    """
    t1 = np.zeros((5, H, W), dtype=np.float32)
    t2 = np.zeros((5, H, W), dtype=np.float32)

    # Background: dense forest
    _fill_background(t1, "vegetation")
    _fill_background(t2, "vegetation")

    # T2 changes: three logging clearance patches
    patches = [
        (slice(20, 80),  slice(30, 110)),   # large north-west clearance
        (slice(100, 150), slice(150, 220)),  # central-east strip
        (slice(180, 230), slice(60, 130)),   # south-west clearance
    ]
    for row_sl, col_sl in patches:
        _apply_patch(t2, row_sl, col_sl, "bare_soil")

    # T2: logging road tracks (thin horizontal lines)
    _apply_patch(t2, slice(95, 100), slice(0, W), "bare_soil")
    _apply_patch(t2, slice(170, 175), slice(40, 200), "bare_soil")

    t1 += _noise((5, H, W), scale=0.01)
    t2 += _noise((5, H, W), scale=0.01)
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)

    cleared_px = sum((s1.stop - s1.start) * (s2.stop - s2.start) for s1, s2 in patches)
    area_m2    = cleared_px * (GSD_M ** 2)

    meta: dict[str, Any] = {
        "scenario":     "deforestation",
        "title":        "Selective Logging — Western Ghats Forest Edge",
        "location":     "Western Ghats, Karnataka (synthetic)",
        "timestamp_t1": "2022-12-01",
        "timestamp_t2": "2023-03-15",
        "sensor":       "optical",
        "sensor_label": "Cartosat-2S MSS (synthetic)",
        "query_hint":   "deforestation",
        "expected": {
            "primary_index":    "ndvi",
            "change_direction": "vegetation_loss",
            "approx_area_m2":   area_m2,
        },
        "description": (
            "Simulates three selective logging clearance events plus access tracks "
            "on a forested hillside. Dense tropical forest (NDVI≈0.75) is replaced "
            "by bare soil (NDVI≈0.05) within the cleared patches. "
            "Query this scenario with: 'Where has forest cover decreased?'"
        ),
    }
    return t1, t2, meta


def _make_urban_expansion_scenario() -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Urban Expansion Scenario
    ------------------------
    Context : Peri-urban fringe of Hyderabad — farmland conversion.
    T1       : Agricultural land with scattered settlements (low NDBI).
    T2       : Large new housing colony + industrial estate constructed.

    Expected output:
        change_direction = "urban_growth"
        primary_index    = "ndbi"
    """
    t1 = np.zeros((5, H, W), dtype=np.float32)
    t2 = np.zeros((5, H, W), dtype=np.float32)

    # Background: agricultural cropland (moderate NDVI)
    _fill_background(t1, "vegetation")
    _fill_background(t2, "vegetation")

    # T1: small existing settlement
    _apply_patch(t1, slice(200, 240), slice(200, 245), "urban")

    # T2: large new housing colony (south-east quadrant)
    _apply_patch(t2, slice(140, 256), slice(120, 256), "urban")

    # T2: industrial estate (north-east)
    _apply_patch(t2, slice(20, 90), slice(170, 250), "urban")

    # T2: road network (thin strips)
    _apply_patch(t2, slice(130, 135), slice(0, W), "urban")   # horizontal arterial
    _apply_patch(t2, slice(0, H),  slice(115, 120), "urban")  # vertical collector

    # T1: same small settlement retained
    _apply_patch(t2, slice(200, 240), slice(200, 245), "urban")

    t1 += _noise((5, H, W), scale=0.009)
    t2 += _noise((5, H, W), scale=0.009)
    t1 = np.clip(t1, 0, 1)
    t2 = np.clip(t2, 0, 1)

    new_urban_px = (256 - 140) * (256 - 120) + (90 - 20) * (250 - 170)
    area_m2      = new_urban_px * (GSD_M ** 2)

    meta: dict[str, Any] = {
        "scenario":     "urban_expansion",
        "title":        "Peri-Urban Expansion — Agricultural Land Conversion",
        "location":     "Hyderabad Fringe, Telangana (synthetic)",
        "timestamp_t1": "2021-06-01",
        "timestamp_t2": "2023-11-01",
        "sensor":       "optical",
        "sensor_label": "Cartosat-2S MSS (synthetic)",
        "query_hint":   "urban growth",
        "expected": {
            "primary_index":    "ndbi",
            "change_direction": "urban_growth",
            "approx_area_m2":   area_m2,
        },
        "description": (
            "Simulates rapid peri-urban growth: farmland converted to a large "
            "housing colony (116 px × 136 px) and an industrial estate (70×80 px) "
            "with a new road network between 2021 and 2023. "
            "Query this scenario with: 'Has new urban development occurred here?'"
        ),
    }
    return t1, t2, meta


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------

def _save_geotiff(array: np.ndarray, path: str) -> None:
    """Write a (5, H, W) float32 array as a 5-band GeoTIFF."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "count":     5,
        "height":    H,
        "width":     W,
        "crs":       BASE_CRS,
        "transform": BASE_TRANSFORM,
        "compress":  "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.update_tags(i, name=name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SCENARIOS = {
    "flood":         _make_flood_scenario,
    "deforestation": _make_deforestation_scenario,
    "urban":         _make_urban_expansion_scenario,
}


def generate_scenario(name: str, output_dir: str) -> dict[str, str]:
    """
    Generate a single named scenario and write T1/T2 GeoTIFFs + metadata.

    Parameters
    ----------
    name       : One of {"flood", "deforestation", "urban"}.
    output_dir : Directory to write files into (created if missing).

    Returns
    -------
    dict with paths to t1_path, t2_path, meta_path.
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Choose from {list(SCENARIOS)}")

    t1, t2, meta = SCENARIOS[name]()

    scenario_dir = os.path.join(output_dir, name)
    os.makedirs(scenario_dir, exist_ok=True)

    t1_path   = os.path.join(scenario_dir, "T1.tif")
    t2_path   = os.path.join(scenario_dir, "T2.tif")
    meta_path = os.path.join(scenario_dir, "scenario_meta.json")

    _save_geotiff(t1, t1_path)
    _save_geotiff(t2, t2_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{name}] T1 -> {t1_path}")
    print(f"  [{name}] T2 -> {t2_path}")
    print(f"  [{name}] meta -> {meta_path}")

    return {
        "scenario":  name,
        "t1_path":   t1_path,
        "t2_path":   t2_path,
        "meta_path": meta_path,
    }


def generate_all_scenarios(output_dir: str = "./demo_data") -> list[dict[str, str]]:
    """
    Generate all three demo scenarios into *output_dir*.
    Returns a list of path dicts (one per scenario).
    """
    print(f"\nSatQuery AI — Demo Scenario Generator")
    print(f"Output directory: {os.path.abspath(output_dir)}\n")

    results = []
    for name in SCENARIOS:
        paths = generate_scenario(name, output_dir)
        results.append(paths)

    summary_path = os.path.join(output_dir, "scenarios_index.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll scenarios generated -> index: {summary_path}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic GeoTIFF demo scenarios for SatQuery AI."
    )
    parser.add_argument(
        "--output-dir", default="./demo_data",
        help="Directory to write scenario GeoTIFFs into (default: ./demo_data)"
    )
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS) + ["all"], default="all",
        help="Which scenario to generate (default: all)"
    )
    args = parser.parse_args()

    if args.scenario == "all":
        generate_all_scenarios(args.output_dir)
    else:
        generate_scenario(args.scenario, args.output_dir)
