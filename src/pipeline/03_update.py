# pipeline/03_update.py
# UHIP Daily Auto-Updater — Landsat-9 LST + UHVI
# Run: python pipeline/03_update.py

import os, datetime
from pathlib import Path
import numpy as np
import pystac_client
import planetary_computer as pc
import stackstac
import rioxarray  # <-- this activates .rio accessor
import xarray as xr

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Delhi bounding box (approx)
DELHI_BBOX = [76.8, 28.4, 77.5, 28.9]

def fetch_latest_landsat():
    print("→ Searching Planetary Computer for latest Landsat-9...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=DELHI_BBOX,
        query={"platform": {"eq": "landsat-9"}},
        sortby=[{"field": "datetime", "direction": "desc"}],
        limit=1,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Landsat-9 scene found")
    item = items[0]
    print(f"✓ Found: {item.id} ({item.datetime.date()})")
    return item

def process_lst(item):
    print("→ Building LST from Band 10...")
    stack = stackstac.stack(
        item,
        assets=["lwir11", "red", "nir08"],
        bounds_latlon=DELHI_BBOX,
        epsg=32643,
        resolution=30,
    ).astype("float32")

    lst_k = stack.sel(band="lwir11").squeeze() * 0.01
    lst_c = lst_k - 273.15
    lst_c = lst_c.where((lst_c > 0) & (lst_c < 60))

    out_path = DATA_DIR / f"LST_Delhi_{datetime.date.today():%Y%m%d}_COG.tif"
    
    # write with rioxarray
    lst_c.rio.write_nodata(-9999, inplace=True)
    lst_c.rio.to_raster(
        out_path,
        driver="COG",
        compress="LZW",
        dtype="float32",
    )
    print(f"✓ LST saved: {out_path}")
    return out_path

def process_uhvi(item):
    print("→ Building UHVI (1-NDVI) from SR bands...")
    stack = stackstac.stack(
        item,
        assets=["red", "nir08"],
        bounds_latlon=DELHI_BBOX,
        epsg=32643,
        resolution=30,
    )
    red = stack.sel(band="red").squeeze()
    nir = stack.sel(band="nir08").squeeze()
    ndvi = (nir - red) / (nir + red)
    uhvi = 1 - ndvi # heat vulnerability
    uhvi = uhvi.clip(0, 1)

    # scale to 0-255 for storage (your API handles it)
    uhvi_uint8 = (uhvi * 255).astype(np.uint8)

    out_path = DATA_DIR / "UHVI_ENHANCED.tif"
    uhvi_uint8.rio.write_crs("EPSG:32643", inplace=True)
    uhvi_uint8.rio.to_raster(
        out_path,
        driver="GTiff",
        compress="LZW",
        dtype="uint8",
        nodata=0,
    )
    print(f"✓ UHVI saved: {out_path}")
    return out_path

if __name__ == "__main__":
    item = fetch_latest_landsat()
    lst_path = process_lst(item)
    uhvi_path = process_uhvi(item)

    # update API pointers (symlink latest)
    latest_lst = DATA_DIR / "LST_Delhi_20260706_COG.tif"
    if latest_lst.exists():
        latest_lst.unlink()
    os.symlink(lst_path.name, latest_lst)

    print("\n✅ Update complete. Restart uvicorn to serve fresh data.")