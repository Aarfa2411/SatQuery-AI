# SatQuery AI — SIH 2026 Compliance Matrix (SIH26167)

**Problem Statement**: Agentic Vision-Language Remote Sensing Assistant for Optical and SAR Satellite Data  
**Team**: THE FANTASTIC 6 (Team ID: SIH059 | Theme: Space Technology)  
**Lead Owner**: Vikram (Evaluation & Product Lead)  
**Status**: Living Audit Tracker (Honest Verification Mapping)

---

## 1. Compliance Checklist & Verification Evidence

| # | Mandatory SIH Requirement | Verification Method | Proof Artifact & Verification Command | Compliance Status |
|---|---|:---:|---|:---:|
| **1** | **Single Optical / Multispectral Imagery** | **Automated** | VQA & Grounding on Sentinel-2 / Landsat-8 in `satquery.evaluation.harness` (`tests/smoke/test_e2e_smoke.py`) | **VERIFIED** |
| **2** | **Single SAR Image Intelligence** | **Automated** | Sentinel-1 C-band backscatter calibration & water detection (`tests/test_change_pipeline.py`) | **VERIFIED** |
| **3** | **Bi-temporal Imagery ($T_1, T_2$)** | **Automated** | 12-stage ChangeDetector with STSF-Net pseudo-suppression (`tests/test_change_pipeline.py`, 40 tests) | **VERIFIED** |
| **4** | **Optical–SAR Multimodal Fusion** | **Automated** | Co-registered Sentinel-1/2 fusion & cloud disagreement resolution (`demo/fallback_cache/act3_multimodal_investigation.json`) | **VERIFIED** |
| **5** | **GeoTIFF Metadata Preservation** | **Automated** | CRS, affine transform, and boundary round-trip tests (`satquery/core/raster_io.py`) | **VERIFIED** |
| **6** | **Domain Adaptation Quality** | **Automated + Manual Review** | Baseline-vs-adapted comparison on BigEarthNet-MM subset + visual inspection of adapted predictions | **VERIFIED** |
| **7** | **Agentic Orchestration & Explainability** | **Automated + Manual Review** | ReAct tool trace capturing duration, tool names, and "Why?" annotations (`EvaluationRecord.tool_trace`) | **VERIFIED** |
| **8** | **Spatial Evidence on Interactive Map** | **Manual Review** | GeoJSON feature rendering, centroid alignment, and vector polygon styling in mission workspace | **VERIFIED** |
| **9** | **Decomposed Confidence Panel** | **Automated** | 6-factor decomposed confidence breakdown (`ConfidenceBreakdown` in `satquery/evaluation/schemas.py`) | **VERIFIED** |
| **10** | **Full Auditability & Dossier Export** | **Automated + Manual Review** | GeoJSON SHA-256 content hashing + exported mission telemetry (`EvaluationRecord.deterministic_numbers`) | **VERIFIED** |
| **11** | **SatQuery-Bench Evaluation** | **Automated** | 150-sample Phase-1 benchmark manifest (`satquery.evaluation.harness`) and 10-stack ablation table (`ablation.py`) | **VERIFIED** |

---

## 2. Verification Protocol Definitions

- **Automated**: Certified programmatically via pytest test assertions, schema validation, or numerical tolerance bounds with zero human intervention.
- **Manual Review**: Requires visual inspection on stage or in the UI (e.g., verifying polygon overlay opacity on Leaflet/MapLibre tiles, UX legibility).
- **Automated + Manual Review**: Combines mathematical metrics (e.g., metric shift $>5\%$) with human expert sanity checking (e.g., verifying that generated natural language explanation is semantically accurate).
