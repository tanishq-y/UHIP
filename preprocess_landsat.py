"""
preprocess_landsat.py
UHIP - Landsat-8 L2SP preprocessing for Delhi UHI
Run: python preprocess_landsat.py
"""
import rasterio
import numpy as np
from pathlib import Path

# --- CONFIG ---
DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")
DATA_PROC.mkdir(parents=True, exist_ok=True)

# Input files (rename your downloads to these)
B4 = DATA_RAW / "LC08_B4.tif"   # Red
B5 = DATA_RAW / "LC08_B5.tif"   # NIR
B6 = DATA_RAW / "LC08_B6.tif"   # SWIR1
ST = DATA_RAW / "LC08_ST_B10.tif"  # Surface Temp (Kelvin *0.01?)

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        nodata = src.nodata
    arr[arr == nodata] = np.nan
    return arr, profile

print("Reading Landsat bands...")
red, prof = read_band(B4)
nir, _ = read_band(B5)
swir, _ = read_band(B6)
lst_raw, _ = read_band(ST)

# --- NDVI ---
ndvi = (nir - red) / (nir + red + 1e-6)
ndvi = np.clip(ndvi, -1, 1)

# --- NDBI ---
ndbi = (swir - nir) / (swir + nir + 1e-6)
ndbi = np.clip(ndbi, -1, 1)

# --- LST ---
# Landsat Collection 2 L2 ST_B10 is in Kelvin scaled by 0.01
lst_k = lst_raw * 0.01
lst_c = lst_k - 273.15

# --- UHVI (simple index) ---
uhvi = (lst_c * (ndbi + 1)) / (ndvi + 1.01)  # +1 to avoid div0
uhvi = np.clip(uhvi, -50, 100)

def save(arr, name):
    prof_out = prof.copy()
    prof_out.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=-9999)
    arr_out = np.where(np.isnan(arr), -9999, arr).astype(np.float32)
    with rasterio.open(DATA_PROC / name, "w", **prof_out) as dst:
        dst.write(arr_out, 1)
    print("Saved", name)

save(ndvi, "NDVI.tif")
save(ndbi, "NDBI.tif")
save(lst_c, "LST_C.tif")
save(uhvi, "UHVI.tif")

print("\nDone! Check data/processed/")
