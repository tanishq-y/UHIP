import rasterio as rio, numpy as np
from pathlib import Path
from rasterio.warp import reproject, Resampling

PROC = Path("data/processed")
RAW = Path("data/raw/landsat")

def load(path, ref_profile=None, ref_shape=None):
    with rio.open(path) as src:
        arr = src.read(1).astype("float32")
        if ref_profile is not None:
            dst = np.empty(ref_shape, dtype="float32")
            reproject(arr, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                      dst_width=ref_shape[1], dst_height=ref_shape[0],
                      resampling=Resampling.bilinear, src_nodata=src.nodata, dst_nodata=np.nan)
            return dst
        return arr, src.profile

lst, prof = load(PROC/"LST_Celsius.tif")
h,w = lst.shape

# NDVI / NDBI from your existing files but force correct range
ndvi,_ = load(PROC/"NDVI.tif", prof, (h,w))
ndbi,_ = load(PROC/"NDBI.tif", prof, (h,w))
build,_ = load(PROC/"BUILD_DENSITY.tif", prof, (h,w))

# fix scaling if NDVI is still DN
ndvi = np.clip(ndvi, -1, 1)
ndbi = np.clip(ndbi, -1, 1)

# TRUE UHVI
mean_lst = np.nanmean(lst)
uhvi = (lst - mean_lst) / mean_lst
uhvi = np.clip(uhvi, -0.4, 0.8) # keep physical range

print(f"LST mean {mean_lst:.1f}C | Lodhi should be < mean, ITO > mean")
print(f"UHVI {np.nanmin(uhvi):.2f} to {np.nanmax(uhvi):.2f}")

prof.update(dtype="float32", nodata=np.nan, compress="lzw")
for name, arr in [("UHVI_FINAL",uhvi), ("NDVI",ndvi), ("NDBI",ndbi), ("BUILD_DENSITY",build)]:
    with rio.open(PROC/f"{name}.tif","w",**prof) as dst:
        dst.write(arr,1)