from pathlib import Path
import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

DATA = Path("data/processed")
files = ["LST_Celsius.tif", "NDVI.tif", "NDBI.tif", "BUILD_DENSITY.tif", "UHVI_FINAL.tif"]

for f in files:
    src = DATA / f
    dst = DATA / f.replace(".tif", "_COG.tif")
    if not src.exists():
        print(f"skip {f}")
        continue
    print(f"COG → {dst.name}")
    cog_translate(
        src,
        dst,
        cog_profiles.get("deflate"),
        overview_level=5,
        overview_resampling="average",
        web_optimized=True
    )
print("✓ All COGs ready")