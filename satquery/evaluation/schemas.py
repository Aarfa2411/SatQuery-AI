"""
Standardized Evaluation Schemas (v0.2)
Defines uniform data models across all specialists, agent planner, and evaluation harness.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ConfidenceBreakdown(BaseModel):
    """
    6-Factor decomposed confidence metric designed for Deep's evidence panel
    and the Act 3 'Why?' moment for SIH 2026 judges.
    """
    input_quality: float = Field(
        1.0, ge=0.0, le=1.0, 
        description="Sensor resolution, cloud cover, radiometric clarity"
    )
    model_confidence: float = Field(
        1.0, ge=0.0, le=1.0, 
        description="Raw softmax / feature separation confidence"
    )
    evidence_agreement: float = Field(
        1.0, ge=0.0, le=1.0, 
        description="Agreement between optical, SAR, and spectral indices"
    )
    geospatial_validity: float = Field(
        1.0, ge=0.0, le=1.0, 
        description="CRS validity, footprint overlap, polygon bounds sanity"
    )
    temporal_validity: float = Field(
        1.0, ge=0.0, le=1.0, 
        description="Sensible T1/T2 baseline and seasonal consistency"
    )
    contradiction_penalty: float = Field(
        0.0, ge=0.0, le=1.0, 
        description="Subtracted penalty if specialists conflict"
    )

    def calculate_composite(self) -> float:
        """
        Compute weighted composite confidence score:
        Weights: input_quality (0.15), model_confidence (0.25), evidence_agreement (0.30),
                 geospatial_validity (0.15), temporal_validity (0.15) minus contradiction_penalty.
        """
        base = (
            0.15 * self.input_quality +
            0.25 * self.model_confidence +
            0.30 * self.evidence_agreement +
            0.15 * self.geospatial_validity +
            0.15 * self.temporal_validity
        )
        composite = max(0.0, min(1.0, base - self.contradiction_penalty))
        return round(composite, 4)


class EvaluationRecord(BaseModel):
    """
    Standard evaluation record representing output of any specialist or agent execution.
    """
    task_id: str = Field(..., description="Unique query or benchmark execution ID")
    schema_version: str = Field("0.2", description="Schema version identifier")
    status: str = Field(
        ...,
        description="Status label: real_model | domain_adapted_model | classical_algorithm | heuristic_fallback | mock | unavailable | failed"
    )
    text_response: str = Field(..., description="Generated natural language explanation/answer")
    deterministic_numbers: Dict[str, float] = Field(
        default_factory=dict,
        description="Key-value pairs of deterministic metrics (e.g. area_ha, change_pct, distance_m)"
    )
    geojson_geometry: Optional[Dict[str, Any]] = Field(
        default=None,
        description="GeoJSON Feature or FeatureCollection containing detected polygons/points"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence (0.0 to 1.0)")
    confidence_breakdown: Optional[ConfidenceBreakdown] = Field(
        default=None,
        description="Optional 6-factor decomposed confidence breakdown"
    )
    execution_time_ms: float = Field(..., ge=0.0, description="Total execution latency in milliseconds")
    tool_trace: List[str] = Field(
        default_factory=list,
        description="Ordered list of tools invoked in this execution"
    )


class BenchmarkItem(BaseModel):
    """
    Single test sample in SatQuery-Bench.
    """
    item_id: str = Field(..., description="Unique sample ID (e.g. SQB-LANDCOVER-001)")
    category: str = Field(..., description="Task category (1 of 10)")
    modality: str = Field(..., description="optical | sar | bitemporal | multimodal")
    input_query: str = Field(..., description="User query / prompt")
    dataset_source: str = Field(..., description="Source dataset (e.g. RSVQA, BigEarthNet-MM, Copernicus)")
    scene_references: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata, scene IDs, or relative file paths (no committed binary blobs)"
    )
    ground_truth: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ground truth targets: text, numbers, classes, or coordinates"
    )


class BenchmarkReport(BaseModel):
    """
    Full summary report of a benchmark evaluation run.
    """
    report_id: str
    schema_version: str = "0.2"
    timestamp: str
    phase: str = "Phase-1 (150 Samples)"
    total_items: int
    executed_items: int
    metrics_summary: Dict[str, float] = Field(default_factory=dict)
    per_category_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    failure_count: int = 0
    passed_dod: bool = False
