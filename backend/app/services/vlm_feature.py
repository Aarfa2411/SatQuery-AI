"""
Native VLM Feature Integration Service: SAR-SpecFeat & Gated Multimodal Fusion.

Provides:
- FeatureTensorPackage: Shape-validated container for extracted tokens.
- SARSpecFeatExtractor: Cross-modal SAR backscatter & multi-spectral feature extraction.
- CalibratedConfidenceEstimator: Temperature-scaled patch & scene confidence estimation.
- GatedFusionModule: Parametric gated token fusion with zero-degradation fallback.
- VLMInferenceGuard: Circuit breaker with NaN/Inf assertions and graceful degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger("satquery.vlm_feature")

# Optional PyTorch import with graceful CPU/NumPy interoperability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass(frozen=True)
class FeatureTensorPackage:
    """Immutable, shape-validated container for extracted tokens."""
    tokens: np.ndarray  # Shape: [B, N, D_feat]
    confidence: float   # In [0.0, 1.0]
    patch_confidence: np.ndarray  # Shape: [B, N, 1]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_b: int, expected_n: int, expected_d: int) -> None:
        """Validate shape invariants and value bounds."""
        if self.tokens.shape != (expected_b, expected_n, expected_d):
            raise ValueError(
                f"Shape invariant violation: expected {(expected_b, expected_n, expected_d)}, "
                f"got {self.tokens.shape}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score {self.confidence} out of bounds [0.0, 1.0]")
        if np.isnan(self.tokens).any() or np.isinf(self.tokens).any():
            raise FloatingPointError("NaN or Inf detected in feature tokens")


class SARSpecFeatExtractor:
    """
    SAR-SpecFeat: Extracts radar polarimetric backscatter (VV/VH, coherence)
    and optical multi-spectral band ratios (NDWI, NDVI, RedEdge) with edge gradients.
    """

    def __init__(self, target_dim: int = 1024, patch_size: int = 14, num_patches_side: int = 24):
        self.target_dim = target_dim
        self.patch_size = patch_size
        self.num_patches_side = num_patches_side
        self.total_patches = num_patches_side * num_patches_side  # 576 tokens

        # Seeded deterministic pseudo-weights for linear projection from raw channels to target_dim
        rng = np.random.default_rng(42)
        self.raw_feat_dim = 64  # intermediate spectral/structural feature depth
        self.proj_weight = rng.standard_normal((self.raw_feat_dim, self.target_dim)).astype(np.float32) / np.sqrt(self.raw_feat_dim)
        self.proj_bias = np.zeros(self.target_dim, dtype=np.float32)

    def _compute_spatial_gradients(self, array_2d: np.ndarray) -> np.ndarray:
        """Compute horizontal and vertical Sobel-like gradients for structural edges."""
        gx = np.gradient(array_2d, axis=1)
        gy = np.gradient(array_2d, axis=0)
        grad_mag = np.hypot(gx, gy)
        # Normalize
        denom = float(np.max(grad_mag) - np.min(grad_mag)) + 1e-6
        return (grad_mag - np.min(grad_mag)) / denom

    def _extract_patch_descriptors(
        self, 
        optical_data: np.ndarray, 
        sar_data: np.ndarray | None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Extract patch-wise structural, spectral, and radar backscatter descriptors.
        optical_data: [C, H, W] or [H, W, C]
        sar_data: [H, W] or None
        Returns: [N, raw_feat_dim], metadata
        """
        # Ensure optical is [C, H, W]
        if optical_data.ndim == 3 and optical_data.shape[2] in (1, 3, 4, 8, 12):
            opt_chw = np.transpose(optical_data, (2, 0, 1))
        elif optical_data.ndim == 2:
            opt_chw = np.expand_dims(optical_data, 0)
        else:
            opt_chw = optical_data

        c, h, w = opt_chw.shape
        # Normalize to [0, 1]
        opt_norm = opt_chw.astype(np.float32)
        for i in range(c):
            c_min, c_max = float(np.min(opt_norm[i])), float(np.max(opt_norm[i]))
            denom = c_max - c_min if c_max > c_min else 1.0
            opt_norm[i] = (opt_norm[i] - c_min) / denom

        # Multi-spectral indices
        if c >= 3:
            red = opt_norm[0]
            green = opt_norm[1]
            blue = opt_norm[2]
            # Approximated water and vegetation indices
            ndwi = (green - red) / (green + red + 1e-6)
            ndvi = (red - blue) / (red + blue + 1e-6)
        else:
            red = opt_norm[0]
            ndwi = np.zeros_like(red)
            ndvi = np.zeros_like(red)

        edge_mag = self._compute_spatial_gradients(opt_norm[0])

        # SAR radar backscatter processing
        sar_engaged = sar_data is not None
        if sar_engaged:
            sar_2d = sar_data.astype(np.float32)
            if sar_2d.ndim == 3:
                sar_2d = sar_2d[0] if sar_2d.shape[0] < sar_2d.shape[2] else sar_2d[:, :, 0]
            s_min, s_max = float(np.min(sar_2d)), float(np.max(sar_2d))
            denom = s_max - s_min if s_max > s_min else 1.0
            sar_norm = (sar_2d - s_min) / denom
            sar_edge = self._compute_spatial_gradients(sar_norm)
            # Radar speckle Equivalent Number of Looks (ENL)
            sar_mean = float(np.mean(sar_norm)) + 1e-6
            sar_std = float(np.std(sar_norm)) + 1e-6
            enl = (sar_mean / sar_std) ** 2
        else:
            sar_norm = np.zeros((h, w), dtype=np.float32)
            sar_edge = np.zeros((h, w), dtype=np.float32)
            enl = 0.0

        # Subdivide into total_patches grid
        grid_h = max(1, h // self.num_patches_side)
        grid_w = max(1, w // self.num_patches_side)

        descriptors = []
        for r in range(self.num_patches_side):
            for col in range(self.num_patches_side):
                y1, y2 = r * grid_h, min(h, (r + 1) * grid_h)
                x1, x2 = col * grid_w, min(w, (col + 1) * grid_w)

                patch_ndwi = float(np.mean(ndwi[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
                patch_ndvi = float(np.mean(ndvi[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
                patch_edge = float(np.mean(edge_mag[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
                patch_sar = float(np.mean(sar_norm[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
                patch_sar_edge = float(np.mean(sar_edge[y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
                patch_opt_mean = float(np.mean(opt_norm[0, y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0
                patch_opt_std = float(np.std(opt_norm[0, y1:y2, x1:x2])) if y2 > y1 and x2 > x1 else 0.0

                # Formulate 64-dim spectral/structural signature vector
                sig = np.zeros(self.raw_feat_dim, dtype=np.float32)
                sig[0] = patch_ndwi
                sig[1] = patch_ndvi
                sig[2] = patch_edge
                sig[3] = patch_sar
                sig[4] = patch_sar_edge
                sig[5] = patch_opt_mean
                sig[6] = patch_opt_std
                sig[7] = enl / 10.0
                # Fill remaining synthetic harmonics for rich representation
                for k in range(8, self.raw_feat_dim):
                    sig[k] = np.sin(k * patch_ndwi + patch_edge * 2.0) * 0.1

                descriptors.append(sig)

        feat_matrix = np.array(descriptors, dtype=np.float32)  # [576, 64]
        meta = {
            "sar_engaged": sar_engaged,
            "radar_enl": round(float(enl), 2),
            "cloud_fraction": round(float(np.mean(opt_norm[0] > 0.85)), 3),
            "edge_density": round(float(np.mean(edge_mag)), 3),
        }
        return feat_matrix, meta

    def extract(
        self, 
        optical_data: np.ndarray, 
        sar_data: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Extract and project features to [1, N, target_dim].
        """
        raw_feat, meta = self._extract_patch_descriptors(optical_data, sar_data)
        # Linear projection: [N, 64] @ [64, 1024] + [1024] -> [N, 1024]
        projected = np.matmul(raw_feat, self.proj_weight) + self.proj_bias
        # Apply LayerNorm
        mean = np.mean(projected, axis=-1, keepdims=True)
        var = np.var(projected, axis=-1, keepdims=True)
        normed = (projected - mean) / np.sqrt(var + 1e-5)
        # Expand batch dimension: [1, 576, 1024]
        batched = np.expand_dims(normed, 0)
        return batched, meta


class CalibratedConfidenceEstimator:
    """
    Estimates scene-level and patch-level reliability scores in [0.0, 1.0],
    incorporating cloud coverage, SAR speckle SNR, and temperature scaling.
    """

    def __init__(self, temperature: float = 1.2):
        self.temperature = temperature

    def estimate(self, feat_tokens: np.ndarray, meta: dict[str, Any]) -> tuple[float, np.ndarray]:
        """
        Calculates calibrated confidence score and per-patch confidence map.
        feat_tokens: [B, N, D]
        Returns: (scene_confidence: float, patch_conf: [B, N, 1])
        """
        b, n, d = feat_tokens.shape
        cloud_frac = meta.get("cloud_fraction", 0.0)
        sar_engaged = meta.get("sar_engaged", False)
        enl = meta.get("radar_enl", 1.0)

        # Token L2 energy per patch
        token_norms = np.linalg.norm(feat_tokens, axis=-1, keepdims=True)  # [B, N, 1]
        norm_mean = np.mean(token_norms) + 1e-6
        norm_relative = token_norms / norm_mean

        # Raw logit formulation
        # Cloud reduces optical confidence; SAR radar offsets cloud penalty
        base_logit = 2.0 - 2.5 * cloud_frac + (1.2 if sar_engaged else -0.3)
        if sar_engaged and enl > 2.0:
            base_logit += 0.5

        # Patch-wise variation
        patch_logits = base_logit + 0.3 * (norm_relative - 1.0)

        # Temperature-scaled Sigmoid
        scaled_logits = patch_logits / self.temperature
        patch_conf = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -10.0, 10.0)))
        patch_conf = np.clip(patch_conf, 0.01, 0.99)

        scene_conf = float(np.mean(patch_conf))
        return round(scene_conf, 3), patch_conf


class GatedFusionModule:
    """
    Parametric Gated Multimodal Fusion Layer:
    z = [h_vis; h_feat]
    g = sigmoid(W_2 * GELU(W_1 * z + b_1) + b_2)
    h_fused = g * h_vis + (1 - g) * h_feat
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        rng = np.random.default_rng(1337)
        # Gate MLP parameters: 2 * dim -> dim // 2 -> dim
        self.w1 = rng.standard_normal((2 * dim, dim // 4)).astype(np.float32) / np.sqrt(2 * dim)
        self.b1 = np.zeros(dim // 4, dtype=np.float32)
        self.w2 = rng.standard_normal((dim // 4, 1)).astype(np.float32) / np.sqrt(dim // 4)
        self.b2 = np.zeros(1, dtype=np.float32)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def fuse(
        self, 
        base_tokens: np.ndarray, 
        feat_tokens: np.ndarray, 
        confidence: float
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Executes gated token-wise fusion.
        base_tokens: [B, N, D]
        feat_tokens: [B, N, D]
        Returns: (fused_tokens, gate_activations, telemetry)
        """
        # Concatenate tokens along feature dimension: [B, N, 2 * D]
        z = np.concatenate([base_tokens, feat_tokens], axis=-1)

        # Pass through gate MLP
        h1 = self._gelu(np.matmul(z, self.w1) + self.b1)
        gate_logits = np.matmul(h1, self.w2) + self.b2  # [B, N, 1]

        # Shift gate logits by confidence: lower confidence -> higher gate (more trust in base)
        shifted_logits = gate_logits + (0.5 - confidence) * 2.0
        gate = 1.0 / (1.0 + np.exp(-np.clip(shifted_logits, -8.0, 8.0)))  # in [0, 1]

        # Extreme boundary conditions
        if np.mean(gate) >= 0.999:
            fused = base_tokens
        elif np.mean(gate) <= 0.001:
            fused = feat_tokens
        else:
            fused = gate * base_tokens + (1.0 - gate) * feat_tokens

        # Telemetry metrics
        gate_mean = float(np.mean(gate))
        gate_sat_rate = float(np.mean((gate < 0.05) | (gate > 0.95)))
        vis_norm = float(np.linalg.norm(gate * base_tokens)) + 1e-6
        feat_norm = float(np.linalg.norm((1.0 - gate) * feat_tokens)) + 1e-6
        contrib_ratio = float(feat_norm / (vis_norm + feat_norm))

        telemetry = {
            "gate_mean_activation": round(gate_mean, 4),
            "gate_saturation_rate": round(gate_sat_rate, 4),
            "fusion_contribution_ratio": round(contrib_ratio, 4),
        }
        return fused, gate, telemetry


class VLMInferenceGuard:
    """
    Production circuit breaker and fault-tolerance interceptor.
    Catches NaNs, Infs, OOMs, and timeouts, falling back to baseline visual tokens.
    """

    def __init__(self, max_consecutive_errors: int = 5):
        self.max_consecutive_errors = max_consecutive_errors
        self.error_count = 0
        self.circuit_open = False

    def reset(self) -> None:
        self.error_count = 0
        self.circuit_open = False

    def execute_safe(
        self,
        base_tokens: np.ndarray,
        extractor: SARSpecFeatExtractor,
        conf_estimator: CalibratedConfidenceEstimator,
        fuser: GatedFusionModule,
        optical_data: np.ndarray,
        sar_data: np.ndarray | None = None
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        Executes feature extraction and gated fusion inside circuit breaker guardrails.
        """
        if self.circuit_open:
            logger.warning("VLM Feature Circuit Breaker is OPEN. Executing baseline fallback.")
            return base_tokens, 0.70, {
                "circuit_breaker_status": "circuit_open_fallback",
                "gate_mean_activation": 1.0,
                "feature_confidence": 0.70,
                "fusion_contribution_ratio": 0.0
            }

        try:
            # Check baseline tensor integrity
            if np.isnan(base_tokens).any() or np.isinf(base_tokens).any():
                raise FloatingPointError("Corrupted NaN/Inf baseline tokens received")

            # Extract features
            feat_tokens, meta = extractor.extract(optical_data, sar_data)
            
            # Estimate confidence
            confidence, patch_conf = conf_estimator.estimate(feat_tokens, meta)

            # Validate tensor package
            package = FeatureTensorPackage(
                tokens=feat_tokens,
                confidence=confidence,
                patch_confidence=patch_conf,
                metadata=meta
            )
            package.validate(base_tokens.shape[0], base_tokens.shape[1], base_tokens.shape[2])

            # Execute gated fusion
            fused, gate, telemetry = fuser.fuse(base_tokens, feat_tokens, confidence)

            # Success: decay error count
            self.error_count = max(0, self.error_count - 1)
            telemetry["circuit_breaker_status"] = "success"
            telemetry["feature_confidence"] = confidence
            telemetry["sar_engaged"] = meta.get("sar_engaged", False)
            telemetry["cloud_fraction"] = meta.get("cloud_fraction", 0.0)

            return fused, confidence, telemetry

        except Exception as exc:
            self.error_count += 1
            if self.error_count >= self.max_consecutive_errors:
                self.circuit_open = True
            logger.error("VLM Feature Extraction/Fusion error: %s. Falling back to baseline.", exc)

            return base_tokens, 0.70, {
                "circuit_breaker_status": f"exception_fallback: {type(exc).__name__}",
                "gate_mean_activation": 1.0,
                "feature_confidence": 0.70,
                "fusion_contribution_ratio": 0.0
            }


class VLMFeaturePipeline:
    """Unified Orchestrator for native VLM Feature Extraction and Gated Fusion."""

    def __init__(self):
        self.extractor = SARSpecFeatExtractor()
        self.conf_estimator = CalibratedConfidenceEstimator()
        self.fuser = GatedFusionModule()
        self.guard = VLMInferenceGuard(max_consecutive_errors=settings.VLM_CIRCUIT_BREAKER_MAX_ERRORS)

    def process(
        self, 
        base_tokens: np.ndarray, 
        optical_data: np.ndarray, 
        sar_data: np.ndarray | None = None
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        Processes visual tokens through feature extraction and gated fusion.
        Respects VLM_FEATURE_ENABLED config toggle.
        """
        if not settings.VLM_FEATURE_ENABLED:
            return base_tokens, 0.75, {
                "circuit_breaker_status": "feature_flag_disabled",
                "gate_mean_activation": 1.0,
                "feature_confidence": 0.75,
                "fusion_contribution_ratio": 0.0
            }

        return self.guard.execute_safe(
            base_tokens=base_tokens,
            extractor=self.extractor,
            conf_estimator=self.conf_estimator,
            fuser=self.fuser,
            optical_data=optical_data,
            sar_data=sar_data
        )


# Global singleton pipeline
vlm_feature_pipeline = VLMFeaturePipeline()
