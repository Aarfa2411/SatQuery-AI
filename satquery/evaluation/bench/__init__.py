"""
SatQuery-Bench Dataset & Manifest Engine
Manages the authoritative 150-sample Phase-1 benchmark across 10 remote sensing task categories.
"""

from .loader import BenchmarkLoader, get_benchmark_manifest

__all__ = ["BenchmarkLoader", "get_benchmark_manifest"]
