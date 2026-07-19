# UHIP — Urban Heat and Vegetation Intelligence Platform (Delhi)

An open-source platform that derives Urban Heat Island (UHI) risk for any coordinate in Delhi from satellite imagery, and serves it via a queryable API.

**Status: Backend complete and validated. Frontend integration in progress.**

---

## 1. Overview

Delhi's Land Surface Temperature (LST) ranged from 25.9°C to 62.0°C in May, with built-up areas measuring 4–8°C hotter than parks and green spaces. UHIP derives LST, NDVI, NDBI, building density, and a composite Urban Heat Vulnerability Index (UHVI) from Landsat 9 satellite imagery, and exposes point-level risk queries through a FastAPI backend.

The goal is to make hyperlocal urban heat risk queryable down to a single lat/lon — useful for urban planning, green infrastructure prioritization, and public health context.

## 2. Architecture

```
Planetary Computer (Landsat 9, Level-2)
   -> preprocess_landsat.py     LST from ST_B10 + QA_PIXEL band
   -> build_indices.py          NDVI, NDBI, BUILD_DENSITY, UHVI
   -> make_cogs.py               Cloud-Optimized GeoTIFF conversion
   -> api.py                    FastAPI: /api/point?lat=&lon=
   -> Frontend                  Click map -> fetch API -> risk popup
```

## 3. Tech Stack

Python · rasterio · numpy · FastAPI · uvicorn · Leaflet.js

## 4. Validated Output

The backend has been validated against known reference points:

| Location | LST (°C) | NDVI | NDBI | Build Density | UHVI | Risk Level |
|---|---|---|---|---|---|---|
| Lodhi Garden | 44.50 | 0.293 | -0.114 | 0.516 | -0.021 | **Low** |
| ITO Crossing | 48.57 | 0.209 | -0.038 | 0.628 | 0.272 | **Very High** |

This is a meaningful sanity check: a well-known green space (Lodhi Garden) and a dense traffic junction (ITO Crossing) return clearly differentiated, physically plausible risk levels — confirming the pipeline behaves correctly rather than just running without errors.

**Risk thresholds:** Low (UHVI < -0.015) · Moderate (< 0.05) · High (< 0.15) · Very High (≥ 0.15)

## 5. API

**`GET /api/point?lat={float}&lon={float}`**

Valid range for Delhi: `lat` 28.4–28.9, `lon` 76.8–77.5

Returns:
```json
{
  "lat": 28.5933, "lon": 77.2219,
  "lst_c": 44.4985, "ndvi": 0.2933, "ndbi": -0.1139,
  "build_density": 0.5157, "uhvi": -0.0209,
  "risk_level": "Low"
}
```

## 6. Project Structure

```
UHIP/
├── api.py                     # FastAPI backend
├── preprocess_landsat.py      # LST derivation from raw Landsat bands
├── build_indices.py           # NDVI, NDBI, building density, UHVI
├── make_cogs.py                # Cloud-Optimized GeoTIFF conversion
├── osm_integration.py         # OpenStreetMap data integration
├── index.html                 # Frontend map (Leaflet)
├── data/
│   ├── osm/                   # OSM-derived data
│   └── processed/             # Final indexed rasters (see Data Setup)
├── notebooks/                 # Exploratory analysis
├── src/                       # Core modules
└── requirements.txt
```

## 7. Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data setup
Processed raster files (LST, NDVI, NDBI, BUILD_DENSITY, UHVI — ~5 files) exceed GitHub's size limit and are hosted separately. Download them into `data/processed/`:
link - https://drive.google.com/drive/folders/1DSv_DiMwR5ny1vonoVwgfFcCsRatdcea?usp=drive_link

```bash
pip install gdown
python download_data.py
```

### Run the API
```bash
uvicorn api:app --reload
```

Health check: `http://localhost:8000/`
Example query: `http://localhost:8000/api/point?lat=28.5933&lon=77.2219`

### Regenerating indices (if source data changes)
```bash
python preprocess_landsat.py   # only if new raw Landsat data is added
python build_indices.py        # recomputes all indices
python make_cogs.py            # converts to Cloud-Optimized GeoTIFF
```

## 8. Known Issues & Fixes

- **`{"detail": "Not Found"}`** — confirm the endpoint is `/api/point`, not `/point`
- **Identical output across different coordinates** — indicates stale COG files; regenerate with `build_indices.py` + `make_cogs.py`
- **Frontend CORS errors** — ensure the backend is running with `CORSMiddleware` enabled (already configured in `api.py`)

## 9. Roadmap

- Time-series analysis (2015–2025) to track UHI trends over time
- Sentinel-2 integration for 10m-resolution NDVI (currently Landsat-derived)
- Population exposure modeling — overlaying risk zones with population density
- Production deployment (Render + TiTiler) for tile-based serving at scale

## 10. Team

Backend & data pipeline: complete and validated.
Frontend: in progress (map-based query interface).
Built as part of the ISRO Bharatiya Antariksh Hackathon 2026 (Problem Statement 1).
