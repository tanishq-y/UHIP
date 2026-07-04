'''
Step 1 – Data download stubs
For hackathon speed, use Google Earth Engine (GEE) or manual download.
This script shows the exact calls you need.
'''
from config import CITY, BOUNDS, DATE_RANGE, DATA_DIR
import os

print(f"=== UHIP Step 1: Download for {CITY} ===")
print("Bounds:", BOUNDS)
print("Date range:", DATE_RANGE)
os.makedirs(DATA_DIR, exist_ok=True)

print("""
MANUAL STEPS (do these now, 15 min):
1. Landsat 8/9 Level-2 (LST):
   - USGS EarthExplorer → select Delhi bbox → Landsat 8-9 C2 L2 → download ST_B10.tif + SR_B4/B5
2. Sentinel-2 L2A:
   - Copernicus Browser → Delhi → April-May 2024 → download B04 (Red), B08 (NIR), B11 (SWIR)
3. ERA5:
   - CDS API → 2m_temperature, relative_humidity → monthly mean for Delhi
4. OSM:
   - https://download.geofabrik.de/asia/india.html → delhi-latest.osm.pbf
5. SRTM DEM:
   - EarthExplorer → SRTM 1 Arc-Second

Save all files into data/raw/
Next: run 02_preprocess.py
""")
