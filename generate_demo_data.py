"""
generate_demo_data.py
UHIP - Generate realistic synthetic raster data for Delhi
This creates demo data so the full pipeline can run without raw satellite imagery.
The spatial patterns mimic real Delhi urban heat patterns.

Run: python generate_demo_data.py
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from scipy.ndimage import gaussian_filter

PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# Delhi extent in UTM 43N (EPSG:32643)
# Approximate Delhi NCT bounds in UTM
west, south, east, north = 720000, 3140000, 740000, 3160000
width, height = 400, 400  # 50m resolution grid
transform = from_bounds(west, south, east, north, width, height)

np.random.seed(42)

print("=" * 60)
print("UHIP - Generating Demo Raster Data for Delhi")
print("=" * 60)

# ─── CREATE REALISTIC SPATIAL PATTERNS ───────────────────────────────────────

# 1. Building density: high in center, low at edges (mimics Delhi CBD)
y, x = np.mgrid[0:height, 0:width]
cx, cy = width // 2, height // 2

# Multiple urban centers (CP, ITO, Karol Bagh)
center1 = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (80 ** 2))
center2 = np.exp(-((x - cx + 60) ** 2 + (y - cy - 40) ** 2) / (50 ** 2))
center3 = np.exp(-((x - cx - 80) ** 2 + (y - cy + 30) ** 2) / (40 ** 2))

build_density = 0.5 * center1 + 0.3 * center2 + 0.2 * center3
build_density += np.random.randn(height, width) * 0.05
build_density = gaussian_filter(build_density, sigma=3)
build_density = np.clip(build_density, 0, 1).astype("float32")

# 2. NDVI: inverse of building density + parks
# Parks: Lodhi Garden area, Ridge Forest
park1 = np.exp(-((x - cx + 30) ** 2 + (y - cy + 50) ** 2) / (25 ** 2))  # Lodhi
park2 = np.exp(-((x - cx - 100) ** 2 + (y - cy - 80) ** 2) / (35 ** 2))  # Ridge
park3 = np.exp(-((x - cx + 10) ** 2 + (y - cy - 20) ** 2) / (15 ** 2))  # India Gate

ndvi = 0.15 + 0.5 * (park1 + park2 + 0.7 * park3) - 0.3 * build_density
ndvi += np.random.randn(height, width) * 0.03
ndvi = gaussian_filter(ndvi, sigma=2)
ndvi = np.clip(ndvi, 0.05, 0.75).astype("float32")

# 3. NDBI: correlated with building density, inverse of NDVI
ndbi = 0.4 * build_density - 0.3 * ndvi + 0.1
ndbi += np.random.randn(height, width) * 0.03
ndbi = gaussian_filter(ndbi, sigma=2)
ndbi = np.clip(ndbi, -0.3, 0.5).astype("float32")

# 4. LST: driven by build density, NDBI, inversely by NDVI
# Delhi summer: 28-48°C range
lst_base = 35.0  # mean temperature
lst = lst_base + 8 * build_density + 5 * ndbi - 10 * ndvi + 3 * (1 - ndvi)
lst += np.random.randn(height, width) * 0.8
lst = gaussian_filter(lst, sigma=2)
lst = np.clip(lst, 28, 48).astype("float32")

# 5. UHVI: (LST - mean) / mean
mean_lst = np.mean(lst)
uhvi = ((lst - mean_lst) / mean_lst).astype("float32")

print(f"\n  Grid: {width}x{height} pixels ({(east-west)/width:.0f}m resolution)")
print(f"  LST range: {lst.min():.1f} - {lst.max():.1f} °C (mean: {mean_lst:.1f}°C)")
print(f"  NDVI range: {ndvi.min():.3f} - {ndvi.max():.3f}")
print(f"  NDBI range: {ndbi.min():.3f} - {ndbi.max():.3f}")
print(f"  Building density: {build_density.min():.3f} - {build_density.max():.3f}")
print(f"  UHVI range: {uhvi.min():.3f} - {uhvi.max():.3f}")

# ─── SAVE AS GEOTIFFS ────────────────────────────────────────────────────────
profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "width": width,
    "height": height,
    "count": 1,
    "crs": "EPSG:32643",
    "transform": transform,
    "nodata": np.nan,
    "compress": "lzw",
}

layers = {
    "LST_Celsius": lst,
    "NDVI": ndvi,
    "NDBI": ndbi,
    "BUILD_DENSITY": build_density,
    "UHVI_FINAL": uhvi,
}

print("\n  Saving rasters...")
for name, arr in layers.items():
    # Save both regular and COG versions
    for suffix in [".tif", "_COG.tif"]:
        path = PROC / f"{name}{suffix}"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
    print(f"    ✓ {name}.tif  ({arr.min():.2f} to {arr.max():.2f})")

print("\n" + "=" * 60)
print("✓ Demo data generated in data/processed/")
print("=" * 60)
print("  Now run: python train_model.py")
