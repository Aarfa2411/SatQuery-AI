"""
satquery.change_detection.models
================================
Model abstraction layer supporting learned transformer architectures (ChangeFormer)
and deterministic classical spectral baselines.
"""

from .base import BaseChangeModel, ChangePrediction, ModelStatus
from .classical_adapter import ClassicalSpectralAdapter
from .changeformer import ChangeFormerAdapter
from .registry import get_change_model, register_model

__all__ = [
    "BaseChangeModel",
    "ChangePrediction",
    "ModelStatus",
    "ClassicalSpectralAdapter",
    "ChangeFormerAdapter",
    "get_change_model",
    "register_model",
]
