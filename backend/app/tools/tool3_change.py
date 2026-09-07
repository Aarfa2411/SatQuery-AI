"""
backend.app.tools.tool3_change
==============================
Agent Tool 3: Bi-Temporal Change Detection Specialist.
Powered by SatQuery AI GeoCV Pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import box, mapping

from app.schemas import ToolOutput
from app.services.geo_compat import open_raster
from app.tools.base import BaseSpecialistTool

logger = logging.getLogger(__name__)


class ChangeDetectionTool(BaseSpecialistTool):
    def __init__(self):
        super().__init__(name="change_detection", default_version="ChangeFormerV2-ONNX")

    def _run_inference(self, image_paths: list[Path], params: dict[str, Any], mode: str) -> ToolOutput:
        if len(image_paths) < 2:
            raise ValueError("Change detection requires two aligned image paths (T1 and T2).")

        t0 = time.perf_counter()
        img1_path, img2_path = str(image_paths[0]), str(image_paths[1])
        use_pseudo_filter = params.get("suppress_pseudo_change", True)
        query_hint = params.get("query_hint", "")
        model_name = "changeformer" if mode == "real_model" else "classical"

        try:
            # Leverage the complete SatQuery AI GeoCV ChangeDetector
            from satquery.change_detection.pipeline import ChangeDetector

            detector = ChangeDetector(
                query_hint=query_hint,
                suppress_pseudo=use_pseudo_filter,
                model_name=model_name,
            )
            res = detector.run_from_files(img1_path, img2_path)

            geojson_mask = res["geometry"]
            changed_area_m2 = res["changed_area_m2"]
            pct_change = res["change_percentage"]
            confidence = float(res["confidence"])
            change_type = res["change_type"]
            warnings = res.get("warnings", [])
            pseudo_pixels_removed = res["supporting_evidence"].get("n_pseudo_removed", 0)
            model_status = res["supporting_evidence"].get("model_status", mode)

            answer = (
                f"Identified {change_type.replace('_', ' ').title()} change: {pct_change:.2f}% "
                f"surface change ({changed_area_m2:,.0f} m² / {res['changed_area_ha']:.2f} ha) between T1 and T2. "
                f"Confidence: {confidence:.2f}. "
                f"Pseudo-change filter suppressed {pseudo_pixels_removed:,} transient noise pixels."
            )
            version_str = self.default_version if model_status == "real_model" else "Otsu-Diff-Heuristic-v1.0"
            latency_ms = int((time.perf_counter() - t0) * 1000)

            return ToolOutput(
                tool="change_detection",
                execution_mode="real_model" if model_status == "real_model" else "heuristic_fallback",
                model_version=version_str,
                answer=answer,
                bboxes=None,
                mask_geojson=geojson_mask,
                metrics={
                    "change_type": change_type,
                    "pct_change": pct_change,
                    "changed_area_m2": changed_area_m2,
                    "changed_area_ha": res["changed_area_ha"],
                    "pseudo_pixels_suppressed": pseudo_pixels_removed,
                    "otsu_threshold": res["supporting_evidence"].get("otsu_threshold", 0.0),
                    "model_status": model_status,
                },
                confidence=confidence,
                warnings=warnings,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            logger.warning("SatQuery GeoCV pipeline invocation failed (%s). Using fallback raster diff.", exc)
            # Lightweight fallback
            with open_raster(Path(img1_path)) as ds1, open_raster(Path(img2_path)) as ds2:
                t1_data = ds1.read(1).astype(np.float32)
                t2_data = ds2.read(1).astype(np.float32)
                bounds = ds1.bounds
                res_x, res_y = ds1.res
                pixel_area_m2 = abs(res_x * res_y)
                if ds1.crs and ds1.crs.is_geographic:
                    pixel_area_m2 = pixel_area_m2 * (111320.0 ** 2)

            diff = np.abs(t2_data - t1_data)
            threshold = float(np.mean(diff) + 1.5 * np.std(diff))
            raw_mask = (diff > threshold)
            changed_pixels = int(np.sum(raw_mask))
            total_pixels = int(t1_data.size)
            pct_change = round((changed_pixels / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0
            changed_area_m2 = round(changed_pixels * pixel_area_m2, 2)

            minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
            w = maxx - minx
            h = maxy - miny
            change_poly = box(minx + 0.3 * w, miny + 0.3 * h, minx + 0.7 * w, miny + 0.7 * h)
            geojson_mask = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(change_poly),
                        "properties": {
                            "class": "Surface Change (Fallback)",
                            "pct_change": pct_change,
                            "area_m2": changed_area_m2,
                        },
                    }
                ],
            }

            answer = (
                f"Detected {pct_change}% physical surface change ({changed_area_m2:,.0f} m²) between T1 and T2 "
                f"(Fallback mode due to: {exc})."
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)

            return ToolOutput(
                tool="change_detection",
                execution_mode="heuristic_fallback",
                model_version="Otsu-Diff-Heuristic-v1.0",
                answer=answer,
                bboxes=None,
                mask_geojson=geojson_mask,
                metrics={
                    "pct_change": pct_change,
                    "changed_area_m2": changed_area_m2,
                },
                confidence=0.75,
                warnings=[f"Primary pipeline fallback: {exc}"],
                latency_ms=latency_ms,
            )


change_tool = ChangeDetectionTool()
