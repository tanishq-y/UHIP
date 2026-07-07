import rasterio as rio
from rasterio.warp import reproject, Resampling
import numpy as np
from pathlib import Path

PROC = Path("data/processed")

print("Loading LST as master grid...")
with rio.open(PROC/"LST_Celsius.tif") as ref:
    lst = ref.read(1).astype("float32")
    profile = ref.profile
    shape, transform, crs = lst.shape, ref.transform, ref.crs

def load(path, resampling=Resampling.bilinear):
    with rio.open(path) as src:
        out = np.empty(shape, dtype="float32")
        reproject(
            source=rio.band(src,1), destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan
        )
    return out

ndvi = load(PROC/"NDVI.tif")
ndbi = load(PROC/"NDBI.tif")
build = load(PROC/"BUILD_DENSITY.tif")  # your existing file

def norm(a):
    a = np.where(np.isnan(a), np.nanmedian(a), a)
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

uhvi = 0.4*norm(lst) + 0.3*norm(ndbi) + 0.2*norm(build) - 0.1*norm(ndvi)
uhvi = np.clip(uhvi, 0, 1)

with rio.open(PROC/"UHVI.tif","w",**profile) as dst:
    dst.write(uhvi,1)

print(f"✓ DONE - LST {np.nanmin(lst):.1f}°C to {np.nanmax(lst):.1f}°C")
print(f"✓ UHVI {np.nanmin(uhvi):.3f} to {np.nanmax(uhvi):.3f}")