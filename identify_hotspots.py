"""
identify_hotspots.py
UHIP - Urban Heat Hotspot Identification
Uses statistical methods (Getis-Ord Gi*) and clustering to identify
discrete heat stress zones and export as GeoJSON.

Run: python identify_hotspots.py
"""
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import uniform_filter, label
from shapely.geometry import shape, mapping
import geopandas as gpd
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
OUTPUT_DIR = Path("outputs/hotspots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Hotspot detection parameters
KERNEL_SIZE = 7         # neighborhood for local statistics
Z_THRESHOLD = 1.96      # 95% confidence for Gi* statistic
MIN_HOTSPOT_PIXELS = 50 # minimum cluster size


# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────
def pick(name):
    cog = PROC / f"{name}_COG.tif"
    tif = PROC / f"{name}.tif"
    return cog if cog.exists() else tif


print("=" * 60)
print("UHIP - Heat Hotspot Identification")
print("=" * 60)

print("\n[1/5] Loading LST and UHVI rasters...")
with rasterio.open(pick("LST_Celsius")) as src:
    lst = src.read(1).astype("float32")
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    if nodata is not None:
        lst[lst == nodata] = np.nan

with rasterio.open(pick("UHVI_FINAL")) as src:
    uhvi = src.read(1).astype("float32")
    uhvi_nodata = src.nodata
    if uhvi_nodata is not None:
        uhvi[uhvi == uhvi_nodata] = np.nan

print(f"  LST: {lst.shape}, range {np.nanmin(lst):.1f} - {np.nanmax(lst):.1f}°C")
print(f"  UHVI: range {np.nanmin(uhvi):.3f} - {np.nanmax(uhvi):.3f}")


# ─── 2. GETIS-ORD Gi* STATISTIC ─────────────────────────────────────────────
print("\n[2/5] Computing Getis-Ord Gi* hot/cold spots...")

# Replace NaN with mean for spatial stats (masked back later)
valid_mask = np.isfinite(lst)
lst_filled = lst.copy()
lst_filled[~valid_mask] = np.nanmean(lst)

# Global statistics
n = valid_mask.sum()
x_bar = np.nanmean(lst)
S = np.nanstd(lst)

# Local sum within kernel
local_sum = uniform_filter(lst_filled, size=KERNEL_SIZE) * (KERNEL_SIZE ** 2)
W = KERNEL_SIZE ** 2  # number of neighbors

# Gi* z-score
numerator = local_sum - x_bar * W
denominator = S * np.sqrt((n * W - W ** 2) / (n - 1))
gi_star = np.where(denominator > 0, numerator / denominator, 0)
gi_star[~valid_mask] = np.nan

print(f"  Gi* z-score range: {np.nanmin(gi_star):.2f} to {np.nanmax(gi_star):.2f}")
print(f"  Hot spots (z > {Z_THRESHOLD}): {(gi_star > Z_THRESHOLD).sum():,} pixels")
print(f"  Cold spots (z < -{Z_THRESHOLD}): {(gi_star < -Z_THRESHOLD).sum():,} pixels")

# Save Gi* raster
gi_path = OUTPUT_DIR / "gi_star_zscore.tif"
out_profile = profile.copy()
out_profile.update(dtype="float32", nodata=np.nan, compress="lzw")
with rasterio.open(gi_path, "w", **out_profile) as dst:
    dst.write(gi_star.astype("float32"), 1)
print(f"  Gi* raster saved → {gi_path}")


# ─── 3. CLUSTER HOTSPOTS ────────────────────────────────────────────────────
print("\n[3/5] Clustering significant hotspot regions...")

# Binary hotspot mask
hotspot_mask = (gi_star > Z_THRESHOLD).astype(np.int32)

# Connected component labeling
labeled_array, num_features = label(hotspot_mask)
print(f"  Raw clusters found: {num_features}")

# Filter by minimum size
cluster_stats = []
for cluster_id in range(1, num_features + 1):
    cluster_pixels = (labeled_array == cluster_id)
    n_pixels = cluster_pixels.sum()

    if n_pixels < MIN_HOTSPOT_PIXELS:
        labeled_array[cluster_pixels] = 0
        continue

    # Statistics for this cluster
    cluster_lst = lst[cluster_pixels]
    cluster_uhvi = uhvi[cluster_pixels]

    cluster_stats.append({
        "cluster_id": int(cluster_id),
        "n_pixels": int(n_pixels),
        "area_km2": round(n_pixels * abs(transform.a * transform.e) / 1e6, 3),
        "mean_lst_c": round(float(np.nanmean(cluster_lst)), 2),
        "max_lst_c": round(float(np.nanmax(cluster_lst)), 2),
        "mean_uhvi": round(float(np.nanmean(cluster_uhvi)), 4),
        "mean_gi_star": round(float(np.nanmean(gi_star[cluster_pixels])), 3),
    })

# Re-label
final_mask = (labeled_array > 0).astype(np.uint8)
labeled_final, n_final = label(final_mask)
print(f"  Significant hotspot clusters (>{MIN_HOTSPOT_PIXELS} px): {len(cluster_stats)}")


# ─── 4. VECTORIZE TO GEOJSON ────────────────────────────────────────────────
print("\n[4/5] Vectorizing hotspots to GeoJSON...")

# Assign risk levels
def assign_risk(mean_lst, mean_uhvi):
    if mean_uhvi > 0.15:
        return "Critical"
    elif mean_uhvi > 0.08:
        return "Very High"
    elif mean_uhvi > 0.02:
        return "High"
    else:
        return "Moderate"


# Use rasterio.features.shapes to vectorize
hotspot_polygons = []
for geom, value in shapes(labeled_final.astype("int32"), transform=transform):
    if value == 0:
        continue
    hotspot_polygons.append(shape(geom))

# Build GeoDataFrame
if hotspot_polygons and cluster_stats:
    # Match polygons to stats (by order)
    n_polys = min(len(hotspot_polygons), len(cluster_stats))
    gdf = gpd.GeoDataFrame(
        cluster_stats[:n_polys],
        geometry=hotspot_polygons[:n_polys],
        crs=crs,
    )

    # Add risk level
    gdf["risk_level"] = gdf.apply(
        lambda row: assign_risk(row["mean_lst_c"], row["mean_uhvi"]), axis=1
    )

    # Convert to WGS84 for web compatibility
    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    # Save as GeoJSON
    geojson_path = OUTPUT_DIR / "heat_hotspots.geojson"
    gdf_wgs84.to_file(geojson_path, driver="GeoJSON")
    print(f"  GeoJSON saved → {geojson_path}")
    print(f"  Total hotspot zones: {len(gdf)}")

    # Also save stats as JSON
    stats_path = OUTPUT_DIR / "hotspot_stats.json"
    stats_export = []
    for _, row in gdf_wgs84.iterrows():
        centroid = row.geometry.centroid
        stats_export.append({
            "cluster_id": row["cluster_id"],
            "centroid_lat": round(centroid.y, 5),
            "centroid_lon": round(centroid.x, 5),
            "area_km2": row["area_km2"],
            "mean_lst_c": row["mean_lst_c"],
            "max_lst_c": row["max_lst_c"],
            "mean_uhvi": row["mean_uhvi"],
            "risk_level": row["risk_level"],
        })

    with open(stats_path, "w") as f:
        json.dump(stats_export, f, indent=2)
    print(f"  Stats JSON saved → {stats_path}")
else:
    print("  WARNING: No hotspot polygons generated. Check data quality.")
    gdf = gpd.GeoDataFrame()


# ─── 5. HEAT STRESS MAP ─────────────────────────────────────────────────────
print("\n[5/5] Generating heat stress classification map...")

# Classify LST into heat stress categories
stress_map = np.full_like(lst, np.nan)
stress_map[valid_mask & (lst < np.nanpercentile(lst, 20))] = 1  # Cool
stress_map[valid_mask & (lst >= np.nanpercentile(lst, 20)) & (lst < np.nanpercentile(lst, 40))] = 2  # Moderate
stress_map[valid_mask & (lst >= np.nanpercentile(lst, 40)) & (lst < np.nanpercentile(lst, 60))] = 3  # Warm
stress_map[valid_mask & (lst >= np.nanpercentile(lst, 60)) & (lst < np.nanpercentile(lst, 80))] = 4  # Hot
stress_map[valid_mask & (lst >= np.nanpercentile(lst, 80))] = 5  # Very Hot

# Save stress map
stress_path = OUTPUT_DIR / "heat_stress_map.tif"
stress_profile = profile.copy()
stress_profile.update(dtype="float32", nodata=np.nan, compress="lzw")
with rasterio.open(stress_path, "w", **stress_profile) as dst:
    dst.write(stress_map.astype("float32"), 1)
print(f"  Heat stress map saved → {stress_path}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gi* map
im1 = axes[0].imshow(gi_star, cmap="RdBu_r", vmin=-4, vmax=4)
axes[0].set_title("Getis-Ord Gi* Z-Score\n(Red = Hotspot, Blue = Coldspot)")
plt.colorbar(im1, ax=axes[0], label="Z-score")
axes[0].axis("off")

# Heat stress categories
cmap = plt.colormaps.get_cmap("RdYlBu_r").resampled(5)
im2 = axes[1].imshow(stress_map, cmap=cmap, vmin=1, vmax=5)
axes[1].set_title("Heat Stress Classification")
cbar = plt.colorbar(im2, ax=axes[1], ticks=[1, 2, 3, 4, 5])
cbar.set_ticklabels(["Cool", "Moderate", "Warm", "Hot", "Very Hot"])
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hotspot_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Visualization saved → {OUTPUT_DIR / 'hotspot_maps.png'}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✓ HOTSPOT IDENTIFICATION COMPLETE")
print("=" * 60)
if len(cluster_stats) > 0:
    print(f"  Identified {len(cluster_stats)} significant heat hotspot zones")
    print(f"\n  Top 5 Hotspots:")
    sorted_stats = sorted(cluster_stats, key=lambda x: x["mean_lst_c"], reverse=True)[:5]
    print(f"  {'#':<4} {'Area km²':>10} {'Mean LST':>10} {'Max LST':>10} {'UHVI':>8}")
    print(f"  {'-'*44}")
    for i, s in enumerate(sorted_stats):
        print(f"  {i+1:<4} {s['area_km2']:>10.3f} {s['mean_lst_c']:>8.1f}°C {s['max_lst_c']:>8.1f}°C {s['mean_uhvi']:>8.4f}")
print(f"\n  Output files in: {OUTPUT_DIR}/")
print("  Next step: python validate_model.py")
