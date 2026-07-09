# download_qa_fix.py - get QA for YOUR exact scene
from pystac_client import Client
import planetary_computer, requests
from pathlib import Path

OUT_DIR = Path("data/raw/landsat")
TARGET = "146040_20240519" # must match your ST file

client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

# search 19th May only, over Delhi, and filter in Python
search = client.search(
    collections=["landsat-c2-l2"],
    datetime="2024-05-19/2024-05-20",
    bbox=[76.8, 28.3, 77.6, 28.9],
    limit=20
)

items = list(search.items())
matches = [it for it in items if TARGET in it.id]

if not matches:
    print("Available on 19th:")
    for it in items:
        print(" ", it.id)
    raise SystemExit(f"No {TARGET} found. Trying 146039...")

item = planetary_computer.sign(matches[0])
print(f"Found exact match: {item.id}")

# Download QA_PIXEL
qa_asset = item.assets["qa_pixel"]
href = qa_asset.href
out_path = OUT_DIR / f"{item.id}_QA_PIXEL.TIF"

print(f"Downloading {out_path.name}...")
with requests.get(href, stream=True) as r:
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for c in r.iter_content(1024*1024):
            f.write(c)

print(f"✓ Saved {out_path} - {out_path.stat().st_size/1e6:.1f} MB")
print("Now you have matching pair:")
print(f" {item.id}_ST_B10.TIF")
print(f" {item.id}_QA_PIXEL.TIF")