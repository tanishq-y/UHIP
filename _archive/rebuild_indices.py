import planetary_computer as pc, pystac_client, rasterio, numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling

RAW = Path("data/raw/landsat")
PROC = Path("data/processed")
PROC.mkdir(exist_ok=True)

# 1. Download SR B4 B5 B6 for same ID you already use
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
search = catalog.search(collections=["landsat-c2-l2"], ids=["LC09_L2SP_146040_20240519_02_T1"])
item = list(search.items())[0]
signed = pc.sign(item)

for band in ["SR_B4","SR_B5","SR_B6"]:
    url = signed.assets[band].href
    out = RAW / f"LC09_146040_20240519_{band}.TIF"
    if out.exists(): continue
    print(f"Downloading {band}...")
    import requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out,"wb") as f:
            for chunk in r.iter_content(1<<20): f.write(chunk)

# 2. Load LST as master
with rasterio.open(PROC/"LST_Celsius.tif") as ref:
    prof = ref.profile
    h,w = ref.shape
    lst = ref.read(1)

def load_sr(path):
    with rasterio.open(path) as src:
        dn = src.read(1).astype("float32")
        sr = dn * 0.0000275 - 0.2 # C2 L2 scale
        dst = np.empty((h,w), dtype="float32")
        reproject(sr, dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=prof["transform"], dst_crs=prof["crs"],
                  dst_width=w, dst_height=h, resampling=Resampling.bilinear,
                  src_nodata=0, dst_nodata=np.nan)
        return dst

print("Computing NDVI / NDBI from SR...")
red = load_sr(RAW/"LC09_146040_20240519_SR_B4.TIF")
nir = load_sr(RAW/"LC09_146040_20240519_SR_B5.TIF")
swir = load_sr(RAW/"LC09_146040_20240519_SR_B6.TIF")

ndvi = (nir - red) / (nir + red + 1e-6)
ndbi = (swir - nir) / (swir + nir + 1e-6)

# Build density = normalized NDBI where built
build = np.clip((ndbi - np.nanmin(ndbi)) / (np.nanmax(ndbi)-np.nanmin(ndbi)+1e-6), 0,1)
build = np.where(ndvi > 0.3, build*0.3, build) # suppress veg

mean_lst = np.nanmean(lst)
uhvi = (lst - mean_lst) / mean_lst

for name, arr in [("NDVI",ndvi),("NDBI",ndbi),("BUILD_DENSITY",build),("UHVI_FINAL",uhvi)]:
    arr = np.where(np.isfinite(lst), arr, np.nan)
    with rasterio.open(PROC/f"{name}.tif","w",**prof) as dst:
        dst.write(arr.astype("float32"),1)
    print(f"✓ {name}: {np.nanmin(arr):.2f} to {np.nanmax(arr):.2f}")

print(f"Mean LST {mean_lst:.1f}C - Lodhi should now be < mean")