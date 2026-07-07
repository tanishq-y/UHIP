import osmnx as ox
import rasterio as rio
from rasterio import features
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

PROC = Path("data/processed")
LST_PATH = PROC / "LST_Celsius.tif"
OUT_PATH = PROC / "building_density.tif"

print("Loading LST reference grid...")
with rio.open(LST_PATH) as ref:
    profile = ref.profile.copy()
    transform = ref.transform
    shape = ref.read(1).shape
    crs = ref.crs

# --- DELHI NCT ONLY (not whole scene) ---
north, south, east, west = 28.88, 28.40, 77.35, 76.84
print(f"Delhi BBOX: {west},{south},{east},{north}")

print("Downloading OSM buildings for Delhi (smaller query)...")
ox.settings.timeout = 300
ox.settings.overpass_settings = '[out:json][timeout:300]'
# use a different server to avoid timeout
ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api/interpreter"

tags = {"building": True}
gdf = ox.features_from_bbox(bbox=(north, south, east, west), tags=tags)
print(f"Found {len(gdf)} buildings")

gdf = gdf.to_crs(crs)

print("Rasterizing...")
building_mask = features.rasterize(
    [(geom, 1) for geom in gdf.geometry if geom is not None],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

print("Computing density...")
density = gaussian_filter(building_mask.astype("float32"), sigma=5)
density = (density - density.min()) / (density.max() - density.min() + 1e-9)

profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
with rio.open(OUT_PATH, "w", **profile) as dst:
    dst.write(density, 1)

print(f"✓ building_density.tif created - shape {shape}")