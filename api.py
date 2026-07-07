from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import rasterio
from rasterio.warp import transform
import numpy as np
from pathlib import Path
import yaml
from functools import lru_cache
from rio_tiler.io import COGReader
from rio_tiler.utils import render
from rio_tiler.errors import TileOutsideBounds
from PIL import Image
import io

# --- Load config ---
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DATA_DIR = Path(CFG["paths"]["data_dir"])
LAYERS = {k: DATA_DIR / v for k, v in CFG["layers"].items()}

app = FastAPI(title="UHIP API v2.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Colormap for UHVI (green → yellow → orange → red) ---
UHVI_CMAP = {
    0: [34, 197, 94, 255], # low
    76: [234, 179, 8, 255], # moderate
    128: [249, 115, 22, 255], # high
    178: [220, 38, 38, 255], # extreme
    255: [127, 29, 29, 255],
}

# Transparent tile for out-of-bounds
_trans = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
_buf = io.BytesIO()
_trans.save(_buf, format="PNG")
TRANSPARENT_PNG = _buf.getvalue()

# --- Cached point sampler ---
@lru_cache(maxsize=512)
def sample_raster(path_str: str, lon: float, lat: float):
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
            if not (src.bounds.left <= x <= src.bounds.right and
                    src.bounds.bottom <= y <= src.bounds.top):
                return None
            val = list(rasterio.sample.sample_gen(src, [(x, y)]))[0][0]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return round(float(val), 4)
    except Exception as e:
        print(f"Sample error {path.name}: {e}")
        return None

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": CFG.get("version", "0.1"),
        "layers": {k: p.exists() for k, p in LAYERS.items()}
    }

@app.get("/api/point")
def point(lat: float, lon: float):
    vals = {name: sample_raster(str(path), lon, lat)
            for name, path in LAYERS.items()}

    uhvi = vals.get("uhvi")
    risk = "Unknown"
    if uhvi is not None:
        if uhvi > 0.7:
            risk = "Extreme"
        elif uhvi > 0.5:
            risk = "High"
        elif uhvi > 0.3:
            risk = "Moderate"
        else:
            risk = "Low"

    return {
        "lat": lat,
        "lon": lon,
        "location": f"{lat:.4f}, {lon:.4f}",
        **vals,
        "risk_level": risk,
        "source": f"UHIP {CFG.get('version', '0.1')}"
    }

@app.get("/api/tile/{z}/{x}/{y}.png")
def tile(z: int, x: int, y: int):
    uhvi_path = LAYERS.get("uhvi")
    if not uhvi_path or not uhvi_path.exists():
        return Response(content=TRANSPARENT_PNG, media_type="image/png")

    try:
        with COGReader(str(uhvi_path)) as cog:
            img = cog.tile(x, y, z, tilesize=256)

            data = img.data[0].astype(np.float32)
            mask = img.mask

            # Stretch UHVI 0.3-0.8 to 0-255 for visibility (Delhi range)
            vmin, vmax = 0.3, 0.8
            scaled = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

            png_bytes = render(scaled, mask=mask, colormap=UHVI_CMAP, img_format="PNG")
            return Response(content=png_bytes, media_type="image/png")

    except TileOutsideBounds:
        return Response(content=TRANSPARENT_PNG, media_type="image/png")
    except Exception as e:
        print(f"Tile error {z}/{x}/{y}: {e}")
        return Response(content=TRANSPARENT_PNG, media_type="image/png")