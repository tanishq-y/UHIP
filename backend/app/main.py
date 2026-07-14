"""
UHIP Backend API - Full Implementation
Serves model predictions, cooling simulations, and hotspot data.

Run: uvicorn backend.app.main:app --reload
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import rasterio
from rasterio.warp import transform as warp_transform
import numpy as np
import joblib
import json

app = FastAPI(title="UHIP API", version="0.2", description="Urban Heat Island Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── PATHS ───────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUTS = Path("outputs")


def pick(name):
    cog = PROC / f"{name}_COG.tif"
    tif = PROC / f"{name}.tif"
    return str(cog if cog.exists() else tif)


# ─── LOAD MODEL (lazy) ──────────────────────────────────────────────────────
_model = None
_meta = None


def get_model():
    global _model, _meta
    if _model is None:
        model_path = MODEL_DIR / "xgb_lst_model.joblib"
        meta_path = MODEL_DIR / "model_metadata.json"
        if model_path.exists():
            _model = joblib.load(model_path)
            _meta = json.load(open(meta_path))
    return _model, _meta


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def sample_raster(raster_path: str, lon: float, lat: float):
    try:
        with rasterio.open(raster_path) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            for val in src.sample([(xs[0], ys[0])]):
                v = float(val[0])
                if src.nodata is not None and np.isclose(v, src.nodata):
                    return None
                if np.isnan(v) or np.isinf(v):
                    return None
                return round(v, 4)
    except Exception:
        return None


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "UHIP API v0.2 running",
        "endpoints": [
            "/api/point?lat=28.6139&lon=77.2090",
            "/api/predict",
            "/api/simulate",
            "/api/hotspots",
            "/api/model-info",
        ],
    }


@app.get("/api/point")
def get_point(lat: float = Query(...), lon: float = Query(...)):
    """Query all layers at a geographic point."""
    lst = sample_raster(pick("LST_Celsius"), lon, lat)
    ndvi = sample_raster(pick("NDVI"), lon, lat)
    ndbi = sample_raster(pick("NDBI"), lon, lat)
    build = sample_raster(pick("BUILD_DENSITY"), lon, lat)
    uhvi = sample_raster(pick("UHVI_FINAL"), lon, lat)

    # Risk classification
    if uhvi is None:
        risk = "Unknown"
    elif uhvi < -0.02:
        risk = "Low"
    elif uhvi < 0.08:
        risk = "Moderate"
    elif uhvi < 0.18:
        risk = "High"
    else:
        risk = "Very High"

    return {
        "lat": lat,
        "lon": lon,
        "lst_c": lst,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "build_density": build,
        "uhvi": uhvi,
        "risk_level": risk,
    }


class PredictRequest(BaseModel):
    ndvi: float
    ndbi: float
    build_density: float


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Predict LST from urban drivers using trained XGBoost model."""
    model, meta = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run train_model.py first.")

    feature_names = meta["features"]
    feat_dict = {
        "ndvi": req.ndvi,
        "ndbi": req.ndbi,
        "build_density": req.build_density,
        "ndvi_sq": req.ndvi ** 2,
        "ndbi_x_build": req.ndbi * req.build_density,
        "veg_deficit": 1.0 - req.ndvi,
    }
    X = np.array([[feat_dict[f] for f in feature_names]])
    predicted_lst = float(model.predict(X)[0])

    return {
        "predicted_lst_c": round(predicted_lst, 2),
        "input_features": feat_dict,
        "model_metrics": meta.get("metrics", {}),
    }


class SimulateRequest(BaseModel):
    lat: float
    lon: float
    intervention: str = "urban_greening"  # urban_greening, cool_roofs, green_roofs, water_bodies
    intensity: float = 1.0  # 0.5 = half intensity, 2.0 = double


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Simulate cooling intervention at a specific location."""
    model, meta = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    # Get current values
    ndvi = sample_raster(pick("NDVI"), req.lon, req.lat)
    ndbi = sample_raster(pick("NDBI"), req.lon, req.lat)
    build = sample_raster(pick("BUILD_DENSITY"), req.lon, req.lat)
    lst_current = sample_raster(pick("LST_Celsius"), req.lon, req.lat)

    if any(v is None for v in [ndvi, ndbi, build]):
        raise HTTPException(status_code=404, detail="Location outside data coverage")

    # Apply intervention
    ndvi_new, ndbi_new, build_new = ndvi, ndbi, build
    intensity = req.intensity

    if req.intervention == "urban_greening":
        ndvi_new = min(1.0, ndvi + 0.20 * intensity)
    elif req.intervention == "cool_roofs":
        ndbi_new = max(-1.0, ndbi - 0.15 * intensity)
    elif req.intervention == "green_roofs":
        ndvi_new = min(1.0, ndvi + 0.15 * intensity)
        ndbi_new = max(-1.0, ndbi - 0.10 * intensity)
    elif req.intervention == "water_bodies":
        ndvi_new = 0.1
        ndbi_new = -0.3
        build_new = 0.0
    else:
        raise HTTPException(status_code=400, detail=f"Unknown intervention: {req.intervention}")

    # Predict baseline
    feature_names = meta["features"]

    def make_features(nv, nb, bd):
        d = {"ndvi": nv, "ndbi": nb, "build_density": bd,
             "ndvi_sq": nv ** 2, "ndbi_x_build": nb * bd, "veg_deficit": 1.0 - nv}
        return np.array([[d[f] for f in feature_names]])

    lst_baseline = float(model.predict(make_features(ndvi, ndbi, build))[0])
    lst_after = float(model.predict(make_features(ndvi_new, ndbi_new, build_new))[0])
    temp_reduction = lst_baseline - lst_after

    return {
        "lat": req.lat,
        "lon": req.lon,
        "intervention": req.intervention,
        "intensity": req.intensity,
        "current_lst_c": lst_current,
        "predicted_baseline_c": round(lst_baseline, 2),
        "predicted_after_c": round(lst_after, 2),
        "temperature_reduction_c": round(temp_reduction, 2),
        "features_before": {"ndvi": ndvi, "ndbi": ndbi, "build_density": build},
        "features_after": {"ndvi": round(ndvi_new, 4), "ndbi": round(ndbi_new, 4), "build_density": round(build_new, 4)},
    }


@app.get("/api/hotspots")
def get_hotspots():
    """Return identified heat hotspot zones as GeoJSON."""
    geojson_path = OUTPUTS / "hotspots" / "heat_hotspots.geojson"
    if not geojson_path.exists():
        raise HTTPException(status_code=503, detail="Hotspots not generated yet. Run identify_hotspots.py first.")

    with open(geojson_path) as f:
        return json.load(f)


@app.get("/api/scenario-results")
def get_scenario_results():
    """Return pre-computed scenario comparison results."""
    results_path = OUTPUTS / "scenarios" / "scenario_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=503, detail="Scenarios not simulated yet. Run simulate_cooling.py first.")

    with open(results_path) as f:
        return json.load(f)


@app.get("/api/model-info")
def get_model_info():
    """Return model metadata and performance metrics."""
    _, meta = get_model()
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    return meta


@app.get("/api/validation")
def get_validation():
    """Return model validation report."""
    report_path = OUTPUTS / "validation" / "validation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=503, detail="Validation not run yet. Run validate_model.py first.")

    with open(report_path) as f:
        return json.load(f)
