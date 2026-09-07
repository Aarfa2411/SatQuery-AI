# SatQuery AI — Earth Observation Data Provenance & Licensing Notes

**SIH Problem Statement**: SIH26167 (ISRO / Space Applications Centre)  
**Lead Owner**: Vikram (Evaluation & Product Lead)  
**Purpose**: Full legal and scientific provenance for all satellite datasets utilized in SatQuery-Bench and live system demonstrations.

---

## 1. Open Access & Licensing Summary

All satellite imagery and benchmark subsets used by SatQuery AI are strictly sourced under recognized **Open Science and Open Government Data licenses**. No restricted-access, proprietary, or uncredited imagery is included in the project repository.

| Satellite / Dataset | Sensor Modality | Primary Provider | Open License / Terms | Repository Usage |
|---|---|---|---|---|
| **Copernicus Sentinel-2** | Optical (Multi-spectral, 10m–20m GSD) | European Space Agency (ESA) / CDSE | **Creative Commons Attribution 4.0 International (CC-BY 4.0)** | Live optical VQA, spectral indices (NDVI, NDWI, NDBI), and bi-temporal change detection. |
| **Copernicus Sentinel-1** | C-band Synthetic Aperture Radar (SAR, 20m GSD) | European Space Agency (ESA) / CDSE | **Creative Commons Attribution 4.0 International (CC-BY 4.0)** | All-weather microwave imaging, flood penetration, and cross-modality fusion. |
| **USGS Landsat-8 / Landsat-9** | Optical & Thermal (OLI/TIRS, 15m–30m GSD) | United States Geological Survey (USGS) | **USGS Public Domain (Open Data Policy)** | Extended multi-decadal temporal baselines and land cover validation. |
| **BigEarthNet-MM** | Multispectral & Multimodal Sentinel-1/2 | TU Berlin / DLR | **Open Database License (ODbL 1.0)** | Scene description and land-cover classification benchmark samples. |
| **RSVQA** | Remote Sensing VQA (Sentinel-2) | EPFL | **Creative Commons Attribution 4.0 (CC-BY 4.0)** | Land-cover VQA and question-answering evaluation pairs. |
| **VRSBench** | Visual Reasoning & Grounding in EO | Wuhan University / Research Consortium | **Creative Commons Attribution-NonCommercial (CC-BY-NC 4.0)** | Geospatial object grounding and bounding box evaluation subsets. |
| **CDVQA** | Change Detection Visual Question Answering | Wuhan University / Open Benchmark | **Creative Commons Attribution-NonCommercial (CC-BY-NC 4.0)** | Bi-temporal change verification and narrative evaluation. |

---

## 2. ISRO / Bhuvan Domestic Sensor Alignment Note

- For the SIH Space Technology theme, SatQuery AI includes pre-calibrated sensor profiles and metadata taggers for **ISRO Cartosat-2S** (optical sub-meter GSD) and **ISRO RISAT-1C** (C-band hybrid polarimetric SAR).
- Sample imagery representing these sensors in demo scenarios is derived from public SIH problem-statement sample releases and ISRO Bhuvan open-access data portals, formatted strictly in compliance with open hackathon research dissemination rules.

---

## 3. Zero-Redistribution Repository Policy

To ensure compliance with third-party redistribution terms:
1. **Manifest Pointers Only**: The SatQuery-Bench repository manifest (`satquery/evaluation/bench/manifest.json`) stores metadata, bounding boxes, and scene identifiers only.
2. **On-Demand Fetch / Local Staging**: Evaluation scripts load raw raster bands from local paths or public CDSE APIs at runtime.
