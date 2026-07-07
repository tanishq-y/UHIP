import rasterio as rio
import numpy as np
from pathlib import Path

RAW = Path("data/raw/landsat")   # <-- fixed path
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# find ST_B10 in landsat folder, any case
st_files = list(RAW.rglob("*ST_B10.tif")) + list(RAW.rglob("*ST_B10.TIF"))

if not st_files:
    print("Files found in data/raw/landsat:")
    for f in RAW.rglob("*"):
        if f.is_file(): print(" ", f.name)
    raise FileNotFoundError("No ST_B10 found")

st_path = sorted(st_files)[-1]
print(f"Using: {st_path}")

with rio.open(st_path) as src:
    st = src.read(1).astype("float32")
    profile = src.profile

    st = np.where(st == 0, np.nan, st)
    
    # Landsat Collection 2 Level-2 Surface Temperature
    lst_kelvin = st * 0.00341802 + 149.0
    lst_celsius = lst_kelvin - 273.15
    print(f"Before any masking: min={np.nanmin(st):.0f}, max={np.nanmax(st):.0f} DN")

    print(f"Raw LST: {np.nanmin(lst_celsius):.1f}°C to {np.nanmax(lst_celsius):.1f}°C")

    # Clip to Delhi realistic
    lst_celsius = np.clip(lst_celsius, 20, 50)

profile.update(dtype="float32", compress="lzw", nodata=np.nan)

out_path = PROC / "LST_Celsius.tif"
with rio.open(out_path, "w", **profile) as dst:
    dst.write(lst_celsius, 1)

print(f"✓ Saved {out_path}")
print(f"✓ Final LST: {np.nanmin(lst_celsius):.1f}°C to {np.nanmax(lst_celsius):.1f}°C")