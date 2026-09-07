"""satquery.change_detection — bi-temporal change analysis pipeline."""

from .pipeline import ChangeDetector
from .indices import extract_index, available_indices, ndvi, ndwi, ndbi, rvi
from .detector import detect_changes
from .morphology import clean_mask, label_components
from .confidence import compute_confidence, confidence_label
from .metrics import compute_area, classify_change_direction, build_text_summary

__all__ = [
    "ChangeDetector",
    "extract_index", "available_indices", "ndvi", "ndwi", "ndbi", "rvi",
    "detect_changes",
    "clean_mask", "label_components",
    "compute_confidence", "confidence_label",
    "compute_area", "classify_change_direction", "build_text_summary",
]
