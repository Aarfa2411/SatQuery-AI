"""
satquery.change_detection.models.classical_adapter
==================================================
Adapter wrapping the classical spectral & SAR difference detector as a
compliant BaseChangeModel with ModelStatus.CLASSICAL_ALGORITHM.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from satquery.core.raster_io import RasterData
from satquery.change_detection.detector import detect_changes
from satquery.change_detection.indices import extract_index
from satquery.change_detection.confidence import compute_confidence
from .base import BaseChangeModel, ChangePrediction, ModelStatus


class ClassicalSpectralAdapter(BaseChangeModel):
    """
    Classical remote-sensing change detection model utilizing spectral
    indices (NDVI, NDWI, NDBI, EVI, RVI), STSF-Net pseudo-change suppression,
    and Otsu thresholding.
    """

    def __init__(self, default_index: str = "ndvi"):
        super().__init__(name="ClassicalSpectralDetector", version="1.0.0")
        self.default_index = default_index

    def is_available(self) -> bool:
        return True

    def predict(
        self,
        t1: RasterData,
        t2: RasterData,
        index_name: str | None = None,
        suppress_pseudo: bool = True,
        **kwargs: Any,
    ) -> ChangePrediction:
        start_time = time.perf_counter()
        target_index = index_name or self.default_index

        # Extract spectral or SAR indices
        t1_idx = extract_index(t1, target_index)
        t2_idx = extract_index(t2, target_index)

        # Detect changes using the 5-step core detector (diff, smooth, stsf, otsu)
        det_result = detect_changes(
            t1_idx,
            t2_idx,
            smooth_sigma=1.0,
            suppress_pseudo_changes=suppress_pseudo,
        )

        mask = det_result["change_mask"]
        diff_smoothed = det_result["suppressed_diff"]
        threshold = det_result["otsu_threshold"]

        # Compute normalized probability map using sigmoid-like scaling around threshold
        # For values below threshold prob < 0.5, above threshold prob >= 0.5
        denom = np.std(diff_smoothed) + 1e-6
        norm_diff = (diff_smoothed - threshold) / denom
        prob_map = 1.0 / (1.0 + np.exp(-2.0 * norm_diff))
        prob_map = np.clip(prob_map, 0.0, 1.0).astype(np.float32)

        # Confidence via Otsu bimodal separation
        conf_score = compute_confidence(diff_smoothed, mask, threshold)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        provenance = {
            "index_used": target_index,
            "otsu_threshold": round(float(threshold), 5),
            "pseudo_suppressed": int(det_result.get("n_pseudo_removed", 0)),
            "bimodal_separation": round(float(conf_score), 4),
            "latency_ms": elapsed_ms,
            "device": "cpu",
        }

        return ChangePrediction(
            change_mask=mask,
            probability_map=prob_map,
            model_name=f"{self.name}-{target_index.upper()}",
            model_status=ModelStatus.CLASSICAL_ALGORITHM,
            confidence=float(conf_score),
            provenance=provenance,
        )
