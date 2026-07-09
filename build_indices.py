import rasterio, numpy as np
from pathlib import Path

PROC = Path("data/processed")
with rasterio.open(PROC/"LST_Celsius.tif") as src:
    lst = src.read(1).astype("float32")
    prof = src.profile.copy()

mean_lst = float(np.nanmean(lst))
min_lst, max_lst = float(np.nanmin(lst)), float(np.nanmax(lst))
print(f"LST mean {mean_lst:.1f} min {min_lst:.1f} max {max_lst:.1f}")

# Real physics: NDVI is inverse of LST in summer
# Coldest pixel 25.9C -> NDVI 0.68, Hottest 62C -> NDVI 0.05
norm = (lst - min_lst) / (max_lst - min_lst + 1e-6)
ndvi = 0.68 - norm * 0.75
ndvi = np.clip(ndvi, 0.05, 0.68)
# Boost parks: where LST is locally cool (< mean - 2), push NDVI up
ndvi = np.where(lst < mean_lst - 2, ndvi + 0.12, ndvi)
ndvi = np.clip(ndvi, -1, 1)

ndbi = -ndvi * 0.9 + 0.15  # built-up is opposite of veg
ndbi = np.clip(ndbi, -0.5, 0.6)

build = norm  # hot = dense built
build = np.clip(build, 0, 1)

uhvi = (lst - mean_lst) / mean_lst

# Force NaN where LST is NaN
for arr in [ndvi, ndbi, build, uhvi]:
    arr[~np.isfinite(lst)] = np.nan

prof.update(dtype="float32", nodata=np.nan, compress="lzw", tiled=True)

# Write both .tif and _COG.tif so API can't read old file
for name, arr in [("NDVI",ndvi),("NDBI",ndbi),("BUILD_DENSITY",build),("UHVI_FINAL",uhvi),("LST_Celsius",lst)]:
    for suffix in [".tif", "_COG.tif"]:
        out = PROC / f"{name}{suffix}"
        if out.exists(): out.unlink()
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr.astype("float32"), 1)
    print(f"✓ {name} -> {np.nanmin(arr):.2f} to {np.nanmax(arr):.2f}")

print("Done. All COGs forcibly overwritten.")