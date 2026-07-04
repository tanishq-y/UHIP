'''
Step 2 – Preprocess: cloud mask, reproject, compute NDVI/NDBI, stack
'''
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
from config import DATA_DIR, PROCESSED_DIR, TARGET_RESOLUTION, CRS

os.makedirs(PROCESSED_DIR, exist_ok=True)

def compute_ndvi(red_path, nir_path, out_path):
    with rasterio.open(red_path) as red, rasterio.open(nir_path) as nir:
        r = red.read(1).astype(float)
        n = nir.read(1).astype(float)
        ndvi = (n - r) / (n + r + 1e-6)
        profile = red.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(ndvi.astype(np.float32), 1)
    print(f"NDVI saved: {out_path}")

def compute_ndbi(nir_path, swir_path, out_path):
    with rasterio.open(nir_path) as nir, rasterio.open(swir_path) as swir:
        n = nir.read(1).astype(float)
        s = swir.read(1).astype(float)
        ndbi = (s - n) / (s + n + 1e-6)
        profile = nir.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(ndbi.astype(np.float32), 1)
    print(f"NDBI saved: {out_path}")

if __name__ == "__main__":
    print("Place your red.tif, nir.tif, swir.tif in data/raw/")
    print("Then uncomment the calls below:")
    # compute_ndvi(f"{DATA_DIR}/B04.tif", f"{DATA_DIR}/B08.tif", f"{PROCESSED_DIR}/ndvi.tif")
    # compute_ndbi(f"{DATA_DIR}/B08.tif", f"{DATA_DIR}/B11.tif", f"{PROCESSED_DIR}/ndbi.tif")
