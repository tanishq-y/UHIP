"""
download_sentinel.py
Downloads Sentinel-2 L2A B04, B08, B11 for Delhi T43RGP
Requires: pip install pystac-client planetary-computer rasterio
Run: python download_sentinel.py
"""
from pystac_client import Client
import planetary_computer
import rasterio
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

print("Connecting to Planetary Computer...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects={"type": "Point", "coordinates": [77.2, 28.65]},  # Delhi
    datetime="2024-04-20/2024-04-30",
    query={"eo:cloud_cover": {"lt": 10}},
)

items = list(search.get_items())
print(f"Found {len(items)} scenes")

# Prefer T43RGP, else fallback to nearby
item = None
for i in items:
    if "T43RGP" in i.id:
        item = i
        break
if not item:
    print("T43RGP not found, using closest tile")
    item = items[0]

print("Selected:", item.id, item.datetime)

for band in ["B04", "B08", "B11"]:
    href = item.assets[band].href
    out_path = OUT / f"S2_{band}.tif"
    print(f"Downloading {band}...")
    with rasterio.open(href) as src:
        profile = src.profile
        data = src.read(1)
        profile.update(compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    print(" ->", out_path)

print("\nDone! You now have S2_B04.tif, S2_B08.tif, S2_B11.tif")
