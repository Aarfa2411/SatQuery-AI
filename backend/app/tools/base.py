import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from app.schemas import ToolOutput
from app.services.output_sanity_checker import output_sanity_checker


class BaseSpecialistTool(ABC):
    def __init__(self, name: str, default_version: str):
        self.name = name
        self.default_version = default_version
        self.tier_available: Literal["gpu", "quantized_cpu", "heuristic_only"] = "heuristic_only"
        self._check_available_tiers()

    def _check_available_tiers(self):
        """Check hardware and available weights (GPU -> Quantized CPU -> Heuristic)."""
        # Checks if CUDA or ONNX runtime is available
        try:
            import torch
            if torch.cuda.is_available():
                self.tier_available = "gpu"
                return
        except ImportError:
            pass
        self.tier_available = "heuristic_only"

    def execute(self, image_paths: list[Path], params: dict[str, Any], force_mode: str = "auto") -> ToolOutput:
        start_time = time.perf_counter()
        
        # Decide execution mode
        if force_mode == "real_model" and self.tier_available == "heuristic_only":
            # Real model requested but unavailable
            raise ValueError(f"Requested real_model execution for {self.name}, but real model weights/GPU are not available.")
        
        mode: Literal["real_model", "heuristic_fallback"] = (
            "real_model" if (force_mode == "real_model" or (force_mode == "auto" and self.tier_available in ("gpu", "quantized_cpu")))
            else "heuristic_fallback"
        )
        
        raw_output = self._run_inference(image_paths, params, mode)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        raw_output.latency_ms = latency_ms

        # GeoValidator output sanitization
        return output_sanity_checker.sanitize_tool_output(raw_output)

    @abstractmethod
    def _run_inference(self, image_paths: list[Path], params: dict[str, Any], mode: str) -> ToolOutput:
        pass
