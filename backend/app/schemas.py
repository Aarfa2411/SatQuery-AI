from typing import Any, Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    label: str
    # Normalized [ymin, xmin, ymax, xmax] in [0.0, 1.0]
    box: list[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized to [0.0, 1.0]")
    score: float = Field(..., ge=0.0, le=1.0)

class ToolOutput(BaseModel):
    tool: Literal["vqa_grounding", "change_detection", "optical_sar_fusion", "plain_mapper"]
    execution_mode: Literal["real_model", "heuristic_fallback"]
    model_version: str = Field(..., description="e.g. 'GeoChat-7B-INT8' or 'Otsu-NDWI-v1.0'")
    answer: str
    bboxes: list[BoundingBox] | None = None
    mask_geojson: dict[str, Any] | None = None  # GeoJSON FeatureCollection
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Note: all ratios/fractions stored as float in [0.0, 1.0], e.g. cloud_fraction = 0.42
    confidence: float = Field(..., ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int

class AgentTraceStep(BaseModel):
    step_number: int
    thought: str
    action: str
    tool_called: str
    why_this_tool: str  # Human-readable justification
    tool_input: dict[str, Any]
    observation_summary: str
    step_confidence: float = Field(..., ge=0.0, le=1.0)
    step_overlay: dict[str, Any] | None = None  # Per-step GeoJSON / BoundingBoxes for interactive trace UI

class QueryRequest(BaseModel):
    session_id: str
    query: str
    force_mode: Literal["auto", "real_model", "heuristic_fallback"] | None = "auto"

class QueryResponse(BaseModel):
    session_id: str
    query: str
    final_answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_breakdown: dict[str, float]
    sensor_calibration_badge: str  # e.g. "Cartosat-2S Calibrated", "Sentinel-2 Baseline"
    execution_mode: Literal["real_model", "heuristic_fallback", "hybrid"]
    trace: list[AgentTraceStep]
    composite_overlays: dict[str, Any]  # Unified final overlay layer (GeoJSON + BBoxes)
    metrics_summary: dict[str, Any]
    run_signature_hash: str  # Deterministic SHA-256 (inputs + query + model_versions + outputs)
    report_tamper_token: str # Signed token (signature_hash + timestamp + session_id)
    generated_at: str

class ImageMeta(BaseModel):
    id: str
    filename: str
    crs: str
    bounds: list[float]  # [minx, miny, maxx, maxy]
    resolution_m: float
    width: int
    height: int
    bands: int
    sensor_type: Literal["optical", "sar", "multispectral", "unknown"]
    timestamp: str | None = None
    checksum_sha256: str

class ValidationResult(BaseModel):
    session_id: str
    valid: bool
    mode: Literal["single_image", "bi_temporal", "optical_sar", "unknown"]
    images: list[ImageMeta]
    overlap_area_m2: float | None = None
    overlap_pct: float | None = None
    is_coregistered: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

class ToolInfo(BaseModel):
    name: str
    description: str
    tier_available: Literal["gpu", "quantized_cpu", "heuristic_only"]
    active_mode: Literal["real_model", "heuristic_fallback"]
    model_version: str

class ExportReportRequest(BaseModel):
    session_id: str
    query_response: QueryResponse | None = None
