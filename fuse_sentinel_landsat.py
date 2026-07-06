"""
fuse_sentinel_landsat.py
UHIP - Senior-level fusion for ISRO Hackathon
- Takes Landsat LST (30m) + Sentinel-2 optical (10/20m)
- Resamples, aligns, and produces enhanced NDVI/NDBI/UHVI
Run: python fuse_sentinel_landsat.py
"""
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RS
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(exist_ok=True)

# Inputs
landsat_ref = PROC / "LST_C.tif"  # use Landsat as reference grid (30m)
s2_b4 = RAW / "S2_B04.tif"
s2_b8 = RAW / "S2_B08.tif"
s2_b11 = RAW / "S2_B11.tif"

print("Loading reference grid from Landsat...")
with rasterio.open(landsat_ref) as ref:
    ref_profile = ref.profile
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)

def resample_to_ref(src_path, out_name):
    print(f"Resampling {src_path.name} -> 30m...")
    with rasterio.open(src_path) as src:
        data = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
    # Save
    out_path = PROC / out_name
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.float32, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    return data

s2_red = resample_to_ref(s2_b4, "S2_B04_30m.tif")
s2_nir = resample_to_ref(s2_b8, "S2_B08_30m.tif")
s2_swir = resample_to_ref(s2_b11, "S2_B11_30m.tif")

# Load Landsat LST
with rasterio.open(PROC / "LST_C.tif") as src:
    lst = src.read(1).astype(np.float32)

# Compute enhanced indices (Sentinel optical)
ndvi_s2 = (s2_nir - s2_red) / (s2_nir + s2_red + 1e-6)
ndbi_s2 = (s2_swir - s2_nir) / (s2_swir + s2_nir + 1e-6)

ndvi_s2 = np.clip(ndvi_s2, -1, 1)
ndbi_s2 = np.clip(ndbi_s2, -1, 1)

# Enhanced UHVI
uhvi_enh = (lst * (ndbi_s2 + 1)) / (ndvi_s2 + 1.01)

# Save
def save(arr, name):
    profile = ref_profile.copy()
    profile.update(dtype=rasterio.float32, compress="lzw", nodata=-9999)
    arr_out = np.where(np.isnan(arr), -9999, arr)
    with rasterio.open(PROC / name, "w", **profile) as dst:
        dst.write(arr_out.astype(np.float32), 1)
    print("Saved", name)

save(ndvi_s2, "NDVI_S2.tif")
save(ndbi_s2, "NDBI_S2.tif")
save(uhvi_enh, "UHVI_ENHANCED.tif")

print("\nFusion complete. You now have:")
print("- NDVI_S2.tif (10m->30m, sharper vegetation)")
print("- NDBI_S2.tif (sharper built-up)")
print("- UHVI_ENHANCED.tif (Landsat thermal + Sentinel optical)")
print("\nLoad UHVI_ENHANCED.tif in QGIS, apply 'RdYlBu_r' colormap.")
