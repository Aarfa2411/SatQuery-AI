"""
Benchmark Loader Engine for SatQuery-Bench Phase-1 (150 Samples)
Provides access to metadata, scene references, and ground truth annotations
with stratified subset sampling for rapid ablation studies.
"""

import os
import json
from typing import List, Dict, Any, Optional
from satquery.evaluation.schemas import BenchmarkItem


MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.json")


def get_benchmark_manifest() -> Dict[str, Any]:
    """Loads raw manifest JSON."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class BenchmarkLoader:
    """
    Manages loading, iteration, and stratified sampling of SatQuery-Bench items.
    """

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest_path = manifest_path or MANIFEST_PATH
        self.manifest_data = get_benchmark_manifest()
        self._items: List[BenchmarkItem] = [
            BenchmarkItem(**item) for item in self.manifest_data.get("items", [])
        ]

    def total_count(self) -> int:
        return len(self._items)

    def get_all(self) -> List[BenchmarkItem]:
        """Returns all 150 benchmark items."""
        return list(self._items)

    def filter_by_category(self, category: str) -> List[BenchmarkItem]:
        """Returns items matching a specific category (e.g. 'temporal_change')."""
        return [item for item in self._items if item.category == category]

    def filter_by_modality(self, modality: str) -> List[BenchmarkItem]:
        """Returns items matching optical, sar, bitemporal, or multimodal."""
        return [item for item in self._items if item.modality == modality]

    def get_stratified_subset(self, samples_per_category: int = 3) -> List[BenchmarkItem]:
        """
        Returns a balanced stratified subset across all categories (e.g. 3 * 10 = 30 samples).
        Critique 5 fix: Used for the 10-stack ablation study to avoid Days 10-11 execution bottlenecks.
        """
        subset = []
        categories = sorted(list({item.category for item in self._items}))
        for cat in categories:
            cat_items = self.filter_by_category(cat)
            subset.extend(cat_items[:samples_per_category])
        return subset
