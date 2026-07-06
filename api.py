from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import numpy as np
from datetime import datetime
from pathlib import Path
import glob

app = FastAPI(
    title="UHIP Backend",
    description="ISRO Hackathon - Delhi Urban Heat Island API",
    version="1.1"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
)

# --- AUTO-PICK LATEST RASTERS ---
data_dir = Path("data/processed")
latest_lst_files = sorted(data_dir.glob("LST_Delhi_*_COG.tif"))
if not latest_lst_files:
    raise RuntimeError("No LST files found in data/processed/")

latest_lst = latest_lst_files[-1] # newest by filename date
uhvi_path = data_dir / "UHVI_ENHANCED.tif"

RASTERS = {
    "lst": {
        "path": str(latest_lst),
        "name": "Land Surface Temperature"
    },
    "uhvi": {
        "path": str(uhvi_path),
        "name": "Urban Heat Vulnerability Index"
    }
}

print(f"Loading LST: {latest_lst.name}")
print(f"Loading UHVI: {uhvi_path.name}")

# --- LOAD ONCE AT STARTUP ---
datasets = {}
transformers = {}

for key, meta in RASTERS.items():
    try:
        ds = rasterio.open(meta["path"])
        datasets[key] = ds
        transformers[key] = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        print(f"✓ Loaded {key}: {ds.width}x{ds.height} CRS:{ds.crs}")
    except Exception as e:
        print(f"✗ Failed {key}: {e}")
        datasets[key] = None

def sample_raster(key, lat, lon):
    ds = datasets.get(key)
    if not ds:
        return None
    try:
        x, y = transformers[key].transform(lon, lat)
        row, col = ds.index(x, y)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            return None
        val = ds.read(1, window=Window(col, row, 1, 1))[0, 0]
        if val == ds.nodata or np.isnan(val):
            return None
        val = float(val)

        # --- ROBUST NORMALIZATION FOR UHVI ---
        if key == "uhvi":
            # normalize 0-255 or 0-1000 to 0-1
            if val > 10:
                val = val / 1000 if val > 255 else val / 255
            # invert vegetation to heat vulnerability
            val = 1.0 - val
            val = max(0.0, min(1.0, val))

        return val
    except Exception as e:
        print(f"sample error {key}:", e)
        return None

@app.get("/uhi")
def get_uhi(lat: float, lon: float):
    """Return LST (°C) + UHVI for any lat/lon"""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "Invalid coordinates")

    lst = sample_raster("lst", lat, lon)
    uhvi = sample_raster("uhvi", lat, lon)

    lst_class = (
        "low" if lst and lst < 32 else
        "medium" if lst and lst < 38 else
        "high" if lst else "unknown"
    )

    uhvi_class = (
        "low" if uhvi is not None and uhvi < 0.2 else
        "medium" if uhvi is not None and uhvi < 0.4 else
        "high" if uhvi is not None else "unknown"
    )

    return {
        "query": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "lst_c": round(lst, 1) if lst is not None else None,
            "lst_class": lst_class,
            "uhvi": round(uhvi, 3) if uhvi is not None else None,
            "uhvi_class": uhvi_class,
        },
        "source": {
            "lst": f"Landsat-9 {latest_lst.stem.split('_')[2]}",
            "uhvi": "Landsat-9 + Sentinel-2 fusion"
        },
        "status": "ok" if lst is not None else "outside_coverage"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rasters_loaded": {k: v is not None for k, v in datasets.items()},
        "active_lst": latest_lst.name,
        "coverage": "Delhi NCR"
    }

@app.get("/")
def root():
    return {"message": "UHIP Backend v1.1 - use /uhi?lat=..&lon=.."}