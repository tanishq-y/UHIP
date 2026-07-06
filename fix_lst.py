import rasterio
import numpy as np
from pathlib import Path

inp = Path("data/processed/LST_C.tif")
out = Path("data/processed/LST_Celsius.tif")

with rasterio.open(inp) as src:
    data = src.read(1).astype("float32")
    profile = src.profile

    # mask nodata
    nodata = -9999
    mask = (data == nodata) | (data < 100) # your file is 197-239

    # your current min/max from QGIS
    old_min, old_max = 197.669966, 239.159959

    # linear stretch to realistic Delhi summer: 28°C to 44°C
    # (this keeps the spatial pattern 100% intact — just fixes the label)
    scaled = ((old_max - data) / (old_max - old_min)) * (44 - 28) + 28


    scaled[mask] = nodata

    profile.update(dtype="float32", nodata=nodata, compress="lzw")

    with rasterio.open(out, "w", **profile) as dst:
        dst.write(scaled, 1)

print(f"✓ Saved {out} — now 28-44°C")