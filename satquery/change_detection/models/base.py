"""
satquery.change_detection.models.base
=====================================
Abstract base interfaces, prediction contracts, and model provenance tracking
for SatQuery AI change detection engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from satquery.core.raster_io import RasterData


class ModelStatus(str, Enum):
    """
    Standardized model status tracking per SatQuery AI GeoCV specification:
    - REAL_MODEL: Fully functional deep learning model loaded with authentic weights.
    - DOMAIN_ADAPTED_MODEL: Fine-tuned / domain-adapted model (e.g. for Indian sub-continent/ISRO sensors).
    - CLASSICAL_ALGORITHM: Deterministic spectral/SAR mathematical differencing.
    - HEURISTIC_FALLBACK: Algorithmic fallback when primary learned model fails.
    - MOCK: Synthetic or stubbed output (never treated as production feature).
    - UNAVAILABLE: Model architecture requested but weights/runtime missing.
    - FAILED: Execution failed during inference.
    """
    REAL_MODEL = "real_model"
    DOMAIN_ADAPTED_MODEL = "domain_adapted_model"
    CLASSICAL_ALGORITHM = "classical_algorithm"
    HEURISTIC_FALLBACK = "heuristic_fallback"
    MOCK = "mock"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass
class ChangePrediction:
    """
    Standard output contract produced by any BaseChangeModel implementation.

    Attributes
    ----------
    change_mask : np.ndarray
        2-D boolean mask (H, W) where True = detected surface change.
    probability_map : np.ndarray
        2-D float32 array (H, W) with values in [0.0, 1.0] representing
        change probability for each pixel.
    model_name : str
        Human-readable model identifier (e.g. 'ChangeFormerV2-ONNX', 'ClassicalSpectral-NDVI').
    model_status : ModelStatus
        Honest tracking of the execution mechanism.
    confidence : float
        Overall aggregate prediction confidence in [0.0, 1.0].
    provenance : dict[str, Any]
        Metadata regarding checkpoint path/hash, device, inference latency, and patch configuration.
    """
    change_mask: np.ndarray
    probability_map: np.ndarray
    model_name: str
    model_status: ModelStatus
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_status": self.model_status.value,
            "confidence": round(float(self.confidence), 4),
            "changed_pixels": int(np.sum(self.change_mask)),
            "total_pixels": int(self.change_mask.size),
            "change_percentage": round(float(np.mean(self.change_mask) * 100.0), 3),
            "provenance": self.provenance,
        }


class BaseChangeModel(ABC):
    """
    Abstract interface for all learned and classical change detection models.
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version

    @abstractmethod
    def predict(
        self,
        t1: RasterData,
        t2: RasterData,
        **kwargs: Any,
    ) -> ChangePrediction:
        """
        Execute change detection between bi-temporal rasters T1 and T2.

        Parameters
        ----------
        t1 : RasterData
            First acquisition raster (bands, H, W).
        t2 : RasterData
            Second acquisition raster (bands, H, W). Aligned to T1.
        kwargs : dict
            Optional model-specific runtime hyper-parameters.

        Returns
        -------
        ChangePrediction
            Standardized prediction container.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if model checkpoints and required runtimes are ready."""
        raise NotImplementedError
