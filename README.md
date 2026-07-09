UHIP 0.1 - Urban Heat and Vegetation Intelligence Platform | Delhi
End to end open source platform to query Urban Heat Island risk for any lat lon in Delhi from Landsat 9 satellite.

Status: Backend Complete | Frontend Ready for Integration
Validated: Lodhi Garden -> Low | ITO Crossing -> Very High

1. Overview
Delhi LST ranges 25.9°C to 62.0°C in May. Built up areas are 4 to 8°C hotter than parks. This project derives LST, NDVI, NDBI, BUILD_DENSITY, UHVI from satellite and serves via FastAPI for point level query.

2. Architecture
Planetary Computer (Landsat 9 L2)
-> preprocess_landsat.py -> LST_Celsius.tif [EPSG:32643, 68.46% valid]
-> build_indices.py -> NDVI, NDBI, BUILD_DENSITY, UHVI_FINAL
-> make_cogs.py -> *_COG.tif (Cloud Optimized)
-> api.py -> /api/point?lat=&lon=
-> Leaflet Frontend -> Click -> Fetch API -> Popup Risk

3. Tech Stack
Python, rasterio, numpy, FastAPI, uvicorn, Leaflet.js

4. Folder Structure
UHIP/
├── api.py                 # FastAPI backend - DO NOT CHANGE PORT
├── preprocess_landsat.py  # LST from ST_B10 + QA_PIXEL
├── build_indices.py       # NDVI, NDBI, BUILD, UHVI (was final_fix.py)
├── make_cogs.py           # COG conversion
├── index.html             # Frontend map (for frontend team)
├── data/
│   ├── raw/landsat/       # raw .TIFs (gitignored)
│   └── processed/         # 5 final .tifs (download from Drive)
├── _archive/              # old scripts
└── .gitignore
5. Data Setup - IMPORTANT FOR ALL TEAMMATES
We cannot push .tif to GitHub (100MB limit). Download from Drive.

Drive Link: https://drive.google.com/drive/folders/1DSv_DiMwR5ny1vonoVwgfFcCsRatdcea?usp=sharing
Open Drive link above
Download these 5 files:
LST_Celsius.tif
NDVI.tif
NDBI.tif
BUILD_DENSITY.tif
UHVI_FINAL.tif
Create folder in project root: data/processed/
Paste the 5 files there. Also copy them as *_COG.tif (duplicate and rename) OR run python make_cogs.py
Option B - Auto Download
pip install gdown
python download_data.py
download_data.py content:

import gdown
gdown.download_folder("PASTE_YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE", output="data/processed/", use_cookies=False)

# Test

http://localhost:8000/ -> health check
http://localhost:8000/api/point?lat=28.5933&lon=77.2219 -> should return Low
http://localhost:8000/api/point?lat=28.628&lon=77.241 -> should return Very High
Verified Output
Lodhi Garden:

{
  "lat": 28.5933, "lon": 77.2219,
  "lst_c": 44.4985, "ndvi": 0.2933, "ndbi": -0.1139,
  "build_density": 0.5157, "uhvi": -0.0209,
  "risk_level": "Low"
}
ITO Crossing:

{
  "lat": 28.628, "lon": 77.241,
  "lst_c": 48.5728, "ndvi": 0.2087, "ndbi": -0.0378,
  "build_density": 0.6284, "uhvi": 0.2715,
  "risk_level": "Very High"
}
If you get same output for both points, you have old COGs. Fix:

del data\processed\*COG.tif
python build_indices.py
python make_cogs.py
9. API Docs
GET /api/point?lat=float&lon=float

Params: lat 28.4 to 28.9, lon 76.8 to 77.5 for Delhi
Returns: lst_c, ndvi, ndbi, build_density, uhvi, risk_level

Risk Logic: Low < -0.015, Moderate <0.05, High <0.15, Very High >=0.15

10. How to Regenerate Indices (If You Change LST)
python preprocess_landsat.py   # only if you have new raw Landsat
python build_indices.py        # always overwrites .tif and _COG.tif
python make_cogs.py            # optional if build already wrote COGs
11. Common Issues
{"detail":"Not Found"} -> You are calling wrong URL, use /api/point not /point
Same LST for all points -> CRS bug, update to latest api.py which uses rasterio.warp.transform
Git push fails for .tif -> Normal, data is gitignored, use Drive
Frontend CORS error -> Ensure backend runs with CORSMiddleware (already in api.py)
12. Team Roles
Backend: Complete, validated, handover ready
Frontend: Build Leaflet map, click -> fetch -> popup, risk color badge
Report: Use Low vs Very High screenshots from Section 7
13. Future Work
Time series 2015-2025, Sentinel-2 10m NDVI, population exposure, deploy on Render with TiTiler.

