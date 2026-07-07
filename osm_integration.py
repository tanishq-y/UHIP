# osm_integration.py - v7.1 FIXED URL
from pathlib import Path
import requests
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from scipy.ndimage import uniform_filter

print("UHIP OSM v7.1 - offline PBF")

DATA_DIR = Path("data/processed")
OSM_DIR = Path("data/osm")
OSM_DIR.mkdir(exist_ok=True)
pbf = OSM_DIR / "NewDelhi.osm.pbf"

if not pbf.exists():
    print("Downloading NewDelhi.pbf (~35 MB)...")
    url = "https://download.bbbike.org/osm/bbbike/NewDelhi/NewDelhi.osm.pbf" # FIXED
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(pbf, 'wb') as f:
            for chunk in r.iter_content(1024*1024):
                f.write(chunk)
    print(" ✓ downloaded")

print("Reading buildings...")
buildings = gpd.read_file(pbf, layer='multipolygons')
buildings = buildings[buildings['building'].notna()].copy()
print(f" Found {len(buildings):,} buildings")

buildings = buildings.to_crs("EPSG:32643")

with rasterio.open(DATA_DIR/"LST_C.tif") as src:
    transform, shape, crs = src.transform, src.shape, src.crs

print("Rasterizing...")
raster = rasterize([(geom,1) for geom in buildings.geometry],
                   out_shape=shape, transform=transform, fill=0, dtype='uint8')
density = uniform_filter(raster.astype(np.float32), size=11)

out = DATA_DIR/"BUILD_DENSITY.tif"
with rasterio.open(out, 'w', driver='GTiff', height=shape[0], width=shape[1],
    count=1, dtype='float32', crs=crs, transform=transform, compress='LZW') as dst:
    dst.write(density, 1)

print(f"\n✓✓✓ DONE - {out}")