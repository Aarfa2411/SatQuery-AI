# SatQuery AI — Change Detection Engine & Architecture

**GeoCV Core Technical Documentation**  
*Project*: An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
*Tagline*: Ask Earth. Get Answers.  
*North Star*: **SatQuery AI never answers without showing why.**

---

## 1. Pipeline Overview

The SatQuery AI Change Detection Engine processes bi-temporal satellite acquisitions ($T_1$ and $T_2$) to detect, classify, quantify, and georeference surface changes. It operates through a 12-stage pipeline connecting classical deterministic remote-sensing math with learned transformer models.

```
T1 + T2 Satellite Imagery (GeoTIFF / in-memory raster)
   │
   ▼
Stage 0: GeoValidator Sanity & Consistency Checks
   │ (File integrity, CRS check, spatial overlap, temporal ordering, nodata check)
   ▼
Stage 1: GeoTIFF Ingestion & Normalization
   │ (Windowed read, band extraction, nodata masking)
   ▼
Stage 2: Spatial Co-Registration
   │ (Reproject / resample T2 grid to match T1 exact transform)
   ▼
Stage 3: Spectral Index / Modality Processing
   │ (Auto-detect or user-query directed: NDVI, NDWI, NDBI, EVI, RVI, SAR dB)
   ▼
Stage 4: Absolute / Normalized Differencing
   │ (Δ = |Index(T2) - Index(T1)|)
   ▼
Stage 5: Gaussian Smoothing (Speckle Suppression)
   │ (2D spatial convolution to mitigate high-frequency sensor noise)
   ▼
Stage 6: STSF-Net Pseudo-Change Suppression Filter
   │ (Local variance vs mean signed difference filtering)
   ▼
Stage 7: Otsu Automatic Thresholding & Learned Map Ingestion
   │ (Optimal inter-class variance partition → binary change mask)
   ▼
Stage 8: Morphological Refinement & Segmentation
   │ (Opening, closing, small cluster removal, connected component labeling)
   ▼
Stage 9: Mathematically Grounded Confidence Scoring
   │ (Bimodal histogram separation, valley depth ratio, area penalty)
   ▼
Stage 10: Real-World Quantitative Metrics
   │ (Physical area in m² and hectares, change %, directional classification)
   ▼
Stage 11: Georeferencing & GeoJSON Polygonization
   │ (Pixel coordinates → Raster Affine → Projected CRS → WGS84 EPSG:4326 GeoJSON)
   ▼
Stage 12: Evidence Fusion & Execution Trace Assembly
   │ (Step-by-step audit trail, sensor calibration, warnings, provenance)
   ▼
Downstream Consumer (Tool 3 -> ReAct Agent -> VLM Explanation)
```

---

## 2. Mathematical Formulations

### 2.1 Spectral Indices
- **Normalized Difference Vegetation Index (NDVI)**:
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
- **Normalized Difference Water Index (NDWI)**:
  $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
- **Normalized Difference Built-up Index (NDBI)**:
  $$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}$$
- **Enhanced Vegetation Index (EVI)**:
  $$\text{EVI} = 2.5 \cdot \frac{\text{NIR} - \text{Red}}{\text{NIR} + 6.0 \cdot \text{Red} - 7.5 \cdot \text{Blue} + 1.0}$$
- **Radar Vegetation Index (RVI)** (for Dual-Pol SAR):
  $$\text{RVI} = \frac{4 \cdot \sigma^0_{\text{VH}}}{\sigma^0_{\text{VV}} + \sigma^0_{\text{VH}}}$$

### 2.2 STSF-Net Pseudo-Change Suppression
To prevent spurious detections arising from solar elevation changes, phenological variations, or sensor gain calibration shifts:
1. Compute local spatial standard deviation $\sigma_{T1}$ and $\sigma_{T2}$ in a sliding window $W_{k \times k}$.
2. If local patch variance is high in both temporal acquisitions while mean difference $\mu_\Delta$ remains below a dynamic threshold:
   $$\text{Filter}(x, y) = \mathbb{I}\left(|\Delta(x, y)| > \tau_{\text{Otsu}} \land \text{Sim}(W_{T1}, W_{T2}) < \epsilon\right)$$
3. Suppresses transient agricultural phenology and illumination artifacts without suppressing genuine construction or flooding.

### 2.3 Bimodal Histogram Confidence Scoring
Rather than arbitrary heuristics, confidence is derived from the separation of the bimodal difference histogram:
$$C = \omega \cdot (1.0 - v) \cdot (1.0 - p)$$
Where:
- $\omega = \frac{\sigma_B^2}{\sigma_T^2}$ is the Otsu inter-class variance ratio ($0.0 \le \omega \le 1.0$).
- $v = \frac{H(\tau)}{\min(H(\text{peak}_1), H(\text{peak}_2))}$ is the valley-to-peak depth ratio.
- $p$ is the extreme area imbalance penalty.

---

## 3. Model Abstraction & Provenance Tracking

All models implement `satquery.change_detection.models.BaseChangeModel`.
Every execution tracks an honest `ModelStatus`:
- `REAL_MODEL`: ChangeFormer transformer executing with validated weights.
- `DOMAIN_ADAPTED_MODEL`: Specialized model fine-tuned for regional/ISRO sensors.
- `CLASSICAL_ALGORITHM`: Deterministic spectral/SAR math.
- `HEURISTIC_FALLBACK`: Automatic fallback when learned model weights are unavailable.
- `MOCK`: Synthetic test data (strictly banned from production inference).
- `UNAVAILABLE`: Requested model could not be loaded.
- `FAILED`: Unhandled execution failure.

---

## 4. Geospatial Coordinate Pipeline

The georeferencing pipeline is independently testable:
$$\begin{matrix}
\text{Pixel Coordinates } (r, c) \\
\downarrow \text{ Affine Transform Matrix} \\
\text{Projected Coordinates } (X, Y) \text{ [e.g. UTM EPSG:32643]} \\
\downarrow \text{ PyProj / Rasterio Reprojection} \\
\text{Geographic Coordinates } (\text{Lon}, \text{Lat}) \text{ [WGS84 EPSG:4326]} \\
\downarrow \text{ Shapely Vectorization} \\
\text{Valid GeoJSON FeatureCollection}
\end{matrix}$$
