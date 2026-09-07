"""
Core Evaluation Metrics (P0 Deliverable)
Pure Python / NumPy implementations with zero heavyweight external dependencies.
Covers VQA, classification, segmentation IoU, area error, and confidence calibration.
"""

import re
import string
from typing import List, Union, Tuple
import numpy as np


def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation and excess whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM) metric (1.0 if identical after normalization, else 0.0)."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level precision, recall, and harmonic F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Intersection over Union (IoU) for binary 2D segmentation masks.
    """
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_miou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> float:
    """Mean IoU over a collection of masks."""
    if not pred_masks or len(pred_masks) != len(gt_masks):
        return 0.0
    ious = [compute_mask_iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    return float(np.mean(ious))


def compute_area_error_pct(pred_area: float, gt_area: float) -> float:
    """
    Relative area percentage error: |pred - gt| / gt * 100%.
    """
    if gt_area == 0.0:
        return 0.0 if pred_area == 0.0 else 100.0
    return float(abs(pred_area - gt_area) / abs(gt_area) * 100.0)


def compute_classification_accuracy(preds: List[str], gts: List[str]) -> float:
    """Accuracy across discrete class predictions."""
    if not preds or len(preds) != len(gts):
        return 0.0
    correct = sum(1 for p, g in zip(preds, gts) if str(p).strip().lower() == str(g).strip().lower())
    return float(correct / len(preds))


def compute_expected_calibration_error(
    confidences: List[float], corrects: List[bool], n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) across confidence bins.
    Measures reliability of model confidence.
    """
    if not confidences:
        return 0.0

    confs = np.array(confidences)
    corrs = np.array(corrects, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(corrs[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
