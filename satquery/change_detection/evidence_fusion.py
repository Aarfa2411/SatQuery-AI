"""
satquery.change_detection.evidence_fusion
=========================================
Evidence fusion, cross-corroboration, and discrepancy analysis between learned
deep features (ChangeFormer) and deterministic spectral baselines (NDVI, NDWI, NDBI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from .models.base import DecisionTier


@dataclass
class EvidenceReport:
    """Quantitative evidence corroboration and discrepancy report."""
    learned_classical_iou: float
    learned_supported_fraction: float
    classical_supported_fraction: float
    spectral_contradiction_fraction: float
    alignment_quality: float
    evidence_score: float
    decision_tier: DecisionTier
    diagnostic_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "learned_classical_iou": round(self.learned_classical_iou, 4),
            "learned_supported_fraction": round(self.learned_supported_fraction, 4),
            "classical_supported_fraction": round(self.classical_supported_fraction, 4),
            "spectral_contradiction_fraction": round(self.spectral_contradiction_fraction, 4),
            "alignment_quality": round(self.alignment_quality, 4),
            "evidence_score": round(self.evidence_score, 4),
            "decision_tier": self.decision_tier.value,
            "diagnostic_reason": self.diagnostic_reason,
        }


class EvidenceFusionEngine:
    """
    Fuses multi-spectral indices with learned change probabilities to produce
    auditable decision tiers and corroboration metrics.
    """

    @staticmethod
    def evaluate(
        prob_learned: np.ndarray,
        mask_learned: np.ndarray,
        mask_classical: np.ndarray,
        spectral_diffs: Dict[str, np.ndarray],
        alignment_quality: float = 1.0,
    ) -> EvidenceReport:
        """
        Evaluate agreement, spectral support, and contradiction fractions.
        """
        # 1. Learned-Classical IoU
        inter_lc = np.logical_and(mask_learned, mask_classical).sum()
        union_lc = np.logical_or(mask_learned, mask_classical).sum()
        iou_lc = float(inter_lc / max(1, union_lc))

        # 2. Learned supported fraction: any spectral difference active
        spectral_any = np.zeros_like(mask_learned, dtype=bool)
        for diff in spectral_diffs.values():
            thresh = float(np.mean(diff) + np.std(diff))
            spectral_any |= (diff > thresh)

        n_learned = max(1, int(mask_learned.sum()))
        n_classical = max(1, int(mask_classical.sum()))

        learned_supported = float(np.logical_and(mask_learned, spectral_any).sum() / n_learned)
        classical_supported = float(inter_lc / n_classical)

        # 3. Spectral contradiction: High learned probability (P > 0.8) while all spectral shifts < 0.05
        # and spatial alignment is high (Q_align > 0.85)
        high_prob = prob_learned > 0.8
        all_spectral_zero = np.ones_like(mask_learned, dtype=bool)
        for diff in spectral_diffs.values():
            all_spectral_zero &= (diff < 0.05)

        contradiction_mask = high_prob & all_spectral_zero & (alignment_quality > 0.85)
        contra_fraction = float(contradiction_mask.sum() / max(1, int(high_prob.sum())))

        # 4. Uncalibrated Evidence Score (clearly labeled evidence_score, not calibrated confidence)
        mean_p = float(np.mean(prob_learned[mask_learned])) if mask_learned.any() else 0.0
        evidence_score = float(np.clip(
            0.5 * mean_p + 0.3 * learned_supported + 0.2 * alignment_quality - 0.25 * contra_fraction,
            0.0, 1.0
        ))

        # 5. Multi-tiered Decision Logic
        if contra_fraction > 0.40 and alignment_quality > 0.80:
            tier = DecisionTier.UNCERTAIN
            reason = f"High spectral contradiction ({contra_fraction:.1%}). Flagged for expert review."
        elif learned_supported > 0.60 and evidence_score > 0.70:
            tier = DecisionTier.VERIFIED
            reason = "High agreement between learned features and optical spectral differencing."
        elif evidence_score > 0.45:
            tier = DecisionTier.PROBABLE
            reason = "Probable surface change with moderate corroborating evidence."
        elif mask_learned.sum() == 0 and mask_classical.sum() == 0:
            tier = DecisionTier.REJECTED
            reason = "Null baseline: no significant surface change detected."
        else:
            tier = DecisionTier.REJECTED
            reason = "Weak or transient surface change below decision boundary."

        return EvidenceReport(
            learned_classical_iou=iou_lc,
            learned_supported_fraction=learned_supported,
            classical_supported_fraction=classical_supported,
            spectral_contradiction_fraction=contra_fraction,
            alignment_quality=alignment_quality,
            evidence_score=evidence_score,
            decision_tier=tier,
            diagnostic_reason=reason,
        )
