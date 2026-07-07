# fix_fuse.py
import rasterio, numpy as np
from pathlib import Path
DATA = Path("data/processed")

with rasterio.open(DATA/"LST_Celsius.tif") as s:
    lst = s.read(1).astype('float32')
    prof = s.profile

ndvi = rasterio.open(DATA/"NDVI.tif").read(1)
ndbi = rasterio.open(DATA/"NDBI.tif").read(1)
build = rasterio.open(DATA/"BUILD_DENSITY.tif").read(1)

def norm(a):
    a = np.where(np.isnan(a), np.nan, a)
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

uhvi = 0.4*norm(lst) + 0.3*norm(ndbi) + 0.2*norm(build) - 0.1*norm(ndvi)
uhvi = np.clip(uhvi, 0, 1)

prof.update(dtype='float32', compress='LZW', nodata=np.nan)
with rasterio.open(DATA/"UHVI_FINAL.tif", 'w', **prof) as dst:
    dst.write(uhvi, 1)

print("✓ Fixed UHVI - LST range:", np.nanmin(lst), "to", np.nanmax(lst), "°C")