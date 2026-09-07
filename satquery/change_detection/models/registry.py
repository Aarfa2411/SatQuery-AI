"""
satquery.change_detection.models.registry
=========================================
Factory and registry for instantiating change detection models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import BaseChangeModel
from .changeformer import ChangeFormerAdapter
from .classical_adapter import ClassicalSpectralAdapter

_MODEL_REGISTRY: Dict[str, Callable[..., BaseChangeModel]] = {
    "classical": lambda **kw: ClassicalSpectralAdapter(**kw),
    "classical_spectral": lambda **kw: ClassicalSpectralAdapter(**kw),
    "changeformer": lambda **kw: ChangeFormerAdapter(**kw),
    "changeformer_v2": lambda **kw: ChangeFormerAdapter(**kw),
}


def register_model(name: str, factory: Callable[..., BaseChangeModel]) -> None:
    """Register a new change detection model factory."""
    _MODEL_REGISTRY[name.lower()] = factory


def get_change_model(
    name: str = "classical",
    **kwargs: Any,
) -> BaseChangeModel:
    """
    Get an initialized instance of a change detection model by key.

    Supported keys:
    - 'classical' or 'classical_spectral'
    - 'changeformer' or 'changeformer_v2'
    """
    key = name.lower().strip()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown change model '{name}'. Available: {available}")

    return _MODEL_REGISTRY[key](**kwargs)
