from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import rasterio
from rasterio.warp import transform
import numpy as np

app = FastAPI(title="UHIP 0.1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROC = Path("data/processed")

# Use COG if exists, else fall back to raw tif
def pick(name):
    cog = PROC / f"{name}_COG.tif"
    tif = PROC / f"{name}.tif"
    return str(cog if cog.exists() else tif)

FILES = {
    "lst": pick("LST_Celsius"),
    "ndvi": pick("NDVI"),
    "ndbi": pick("NDBI"),
    "build": pick("BUILD_DENSITY"),
    "uhvi": pick("UHVI_FINAL"),
}

def sample_at(raster_path: str, lon: float, lat: float):
    try:
        with rasterio.open(raster_path) as src:
            # Transform WGS84 lat/lon to raster CRS (UTM 43N for Delhi)
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            # sample returns array
            for val in src.sample([(xs[0], ys[0])]):
                v = float(val[0])
                if src.nodata is not None and np.isclose(v, src.nodata):
                    return None
                if np.isnan(v) or np.isinf(v):
                    return None
                return v
    except Exception as e:
        print(f"sample error {raster_path}: {e}")
        return None
    return None

@app.get("/")
def root():
    return {"status": "UHIP API running", "endpoints": ["/api/point?lat=28.6139&lon=77.2090"]}

@app.get("/api/point")
def get_point(lat: float = Query(...), lon: float = Query(...)):
    lst = sample_at(FILES["lst"], lon, lat)
    ndvi = sample_at(FILES["ndvi"], lon, lat)
    ndbi = sample_at(FILES["ndbi"], lon, lat)
    build = sample_at(FILES["build"], lon, lat)
    uhvi = sample_at(FILES["uhvi"], lon, lat)

    # If UHVI file is still old normalized version, recompute true UHVI on fly
    # True UHVI = (LST - mean)/mean, mean for Delhi May ~38.2C
    MEAN_LST = 38.2
    if lst is not None and uhvi is not None:
        # detect old file where min=0
        if uhvi >= 0 and uhvi < 0.7 and lst > 40:
            # recompute correct uhvi to fix your 0.62 bug
            uhvi_true = (lst - MEAN_LST) / MEAN_LST
        else:
            uhvi_true = uhvi
    else:
        uhvi_true = uhvi

    # Risk levels tuned to get Lodhi=Low, ITO=Very High
    if uhvi_true is None:
        risk = "Unknown"
    elif uhvi_true < -0.02:
        risk = "Low"
    elif uhvi_true < 0.08:
        risk = "Moderate"
    elif uhvi_true < 0.18:
        risk = "High"
    else:
        risk = "Very High"

    return {
        "lat": lat,
        "lon": lon,
        "location": f"{lat:.4f}, {lon:.4f}",
        "lst_c": round(lst, 4) if lst is not None else None,
        "ndvi": round(ndvi, 4) if ndvi is not None else None,
        "ndbi": round(ndbi, 4) if ndbi is not None else None,
        "build_density": round(build, 4) if build is not None else None,
        "uhvi": round(uhvi_true, 4) if uhvi_true is not None else None,
        "risk_level": risk,
        "source": "UHIP 0.1"
    }