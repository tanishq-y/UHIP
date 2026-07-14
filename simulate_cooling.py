"""
simulate_cooling.py
UHIP - Scenario-Based Cooling Intervention Simulator
Uses trained XGBoost model to simulate urban cooling interventions
and quantify temperature reduction for each strategy.

Run: python simulate_cooling.py
Requires: trained model from train_model.py
"""
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from pathlib import Path
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/scenarios")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Intervention parameters (tunable)
SCENARIOS = {
    "urban_greening": {
        "description": "Increase vegetation (NDVI) by 20% in hotspot areas",
        "ndvi_boost": 0.20,      # absolute increase in NDVI
        "target_areas": "high_heat",  # apply only to UHVI > 75th percentile
    },
    "cool_roofs": {
        "description": "Reduce built-up index (NDBI) by 15% via cool roof materials",
        "ndbi_reduction": 0.15,  # absolute decrease in NDBI
        "target_areas": "high_density",  # apply where build_density > 0.6
    },
    "green_roofs": {
        "description": "Combined: reduce NDBI by 10% and increase NDVI by 15%",
        "ndvi_boost": 0.15,
        "ndbi_reduction": 0.10,
        "target_areas": "high_density",
    },
    "water_bodies": {
        "description": "Add water features: set NDVI=0.1, NDBI=-0.3 in selected areas",
        "set_ndvi": 0.1,
        "set_ndbi": -0.3,
        "set_build": 0.0,
        "target_areas": "high_heat_sparse",  # hot areas with low density
    },
    "combined_optimal": {
        "description": "Optimal mix: greening in parks, cool roofs in CBD, water in open hot zones",
        "ndvi_boost": 0.15,
        "ndbi_reduction": 0.12,
        "target_areas": "all_hot",
    },
}


# ─── LOAD MODEL & DATA ──────────────────────────────────────────────────────
def load_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.profile


def pick(name):
    cog = PROC / f"{name}_COG.tif"
    tif = PROC / f"{name}.tif"
    return cog if cog.exists() else tif


print("=" * 60)
print("UHIP - Cooling Scenario Simulator")
print("=" * 60)

# Load model
print("\n[1/5] Loading trained model...")
model = joblib.load(MODEL_DIR / "xgb_lst_model.joblib")
meta = json.load(open(MODEL_DIR / "model_metadata.json"))
feature_names = meta["features"]
print(f"  Model loaded: {meta['model_type']} with features {feature_names}")

# Load rasters
print("\n[2/5] Loading raster layers...")
lst_orig, lst_profile = load_raster(pick("LST_Celsius"))
ndvi_orig, _ = load_raster(pick("NDVI"))
ndbi_orig, _ = load_raster(pick("NDBI"))
build_orig, _ = load_raster(pick("BUILD_DENSITY"))

h, w = lst_orig.shape
print(f"  Grid size: {h} x {w}")
print(f"  Current LST: mean={np.nanmean(lst_orig):.1f}°C, max={np.nanmax(lst_orig):.1f}°C")


# ─── HELPER: BUILD FEATURE MATRIX ───────────────────────────────────────────
def build_features(ndvi, ndbi, build_density):
    """Build feature matrix matching training feature order."""
    feat_dict = {
        "ndvi": ndvi.ravel(),
        "ndbi": ndbi.ravel(),
        "build_density": build_density.ravel(),
        "ndvi_sq": (ndvi ** 2).ravel(),
        "ndbi_x_build": (ndbi * build_density).ravel(),
        "veg_deficit": (1.0 - ndvi).ravel(),
    }
    return np.column_stack([feat_dict[name] for name in feature_names])


# ─── HELPER: PREDICT LST ────────────────────────────────────────────────────
def predict_lst(ndvi, ndbi, build_density):
    """Predict LST for given feature arrays using trained model."""
    X = build_features(ndvi, ndbi, build_density)

    # Valid mask
    valid = np.all(np.isfinite(X), axis=1)

    predicted = np.full(h * w, np.nan, dtype="float32")
    if valid.sum() > 0:
        predicted[valid] = model.predict(X[valid])

    return predicted.reshape(h, w)


# ─── HELPER: IDENTIFY TARGET AREAS ──────────────────────────────────────────
def get_target_mask(target_type):
    """Get boolean mask for intervention target areas."""
    valid = np.isfinite(lst_orig) & np.isfinite(ndvi_orig) & np.isfinite(build_orig)

    if target_type == "high_heat":
        # Top 25% hottest pixels
        threshold = np.nanpercentile(lst_orig, 75)
        return valid & (lst_orig >= threshold)

    elif target_type == "high_density":
        # High building density areas (top 30%)
        threshold = np.nanpercentile(build_orig[valid], 70)
        return valid & (build_orig > threshold)

    elif target_type == "high_heat_sparse":
        # Hot areas with low building density (open land, parking lots)
        threshold = np.nanpercentile(lst_orig, 75)
        build_thresh = np.nanpercentile(build_orig[valid], 40)
        return valid & (lst_orig >= threshold) & (build_orig < build_thresh)

    elif target_type == "all_hot":
        # All pixels above median temperature
        threshold = np.nanpercentile(lst_orig, 50)
        return valid & (lst_orig >= threshold)

    return valid


# ─── 3. BASELINE PREDICTION ─────────────────────────────────────────────────
print("\n[3/5] Computing baseline prediction...")
lst_baseline = predict_lst(ndvi_orig, ndbi_orig, build_orig)
baseline_mean = np.nanmean(lst_baseline)
print(f"  Baseline predicted LST: mean={baseline_mean:.2f}°C")


# ─── 4. RUN SCENARIOS ───────────────────────────────────────────────────────
print("\n[4/5] Simulating cooling scenarios...")
results = {}

for scenario_name, params in SCENARIOS.items():
    print(f"\n  ── {scenario_name} ──")
    print(f"     {params['description']}")

    # Copy original features
    ndvi_mod = ndvi_orig.copy()
    ndbi_mod = ndbi_orig.copy()
    build_mod = build_orig.copy()

    # Get target pixels
    target_mask = get_target_mask(params["target_areas"])
    n_target = target_mask.sum()
    print(f"     Target pixels: {n_target:,} ({100 * n_target / (h * w):.1f}% of grid)")

    # Apply intervention
    if "ndvi_boost" in params:
        ndvi_mod[target_mask] = np.clip(
            ndvi_mod[target_mask] + params["ndvi_boost"], -1, 1
        )
    if "ndbi_reduction" in params:
        ndbi_mod[target_mask] = np.clip(
            ndbi_mod[target_mask] - params["ndbi_reduction"], -1, 1
        )
    if "set_ndvi" in params:
        ndvi_mod[target_mask] = params["set_ndvi"]
    if "set_ndbi" in params:
        ndbi_mod[target_mask] = params["set_ndbi"]
    if "set_build" in params:
        build_mod[target_mask] = params["set_build"]

    # Predict new LST
    lst_scenario = predict_lst(ndvi_mod, ndbi_mod, build_mod)

    # Compute temperature change
    delta = lst_scenario - lst_baseline
    delta_at_target = delta[target_mask]

    if n_target == 0:
        mean_reduction = 0.0
        max_reduction = 0.0
    else:
        mean_reduction = -np.nanmean(delta_at_target)
        max_reduction = -np.nanmin(delta_at_target)
    overall_reduction = -(np.nanmean(lst_scenario) - baseline_mean)

    results[scenario_name] = {
        "description": params["description"],
        "target_pixels": int(n_target),
        "target_pct": round(100 * n_target / (h * w), 1),
        "mean_temp_reduction_target_C": round(float(mean_reduction), 3),
        "max_temp_reduction_C": round(float(max_reduction), 3),
        "overall_city_reduction_C": round(float(overall_reduction), 3),
        "new_mean_lst_C": round(float(np.nanmean(lst_scenario)), 2),
    }

    print(f"     Mean cooling in target area: -{mean_reduction:.2f}°C")
    print(f"     Max cooling:                 -{max_reduction:.2f}°C")
    print(f"     Overall city-wide cooling:   -{overall_reduction:.2f}°C")

    # Save delta map as GeoTIFF
    delta_path = OUTPUT_DIR / f"delta_{scenario_name}.tif"
    profile = lst_profile.copy()
    profile.update(dtype="float32", nodata=np.nan, compress="lzw")
    with rasterio.open(delta_path, "w", **profile) as dst:
        dst.write(delta.astype("float32"), 1)


# ─── 5. OPTIMAL STRATEGY SUMMARY ────────────────────────────────────────────
print("\n[5/5] Generating optimal intervention strategy...")

# Rank scenarios
ranked = sorted(results.items(), key=lambda x: x[1]["mean_temp_reduction_target_C"], reverse=True)

print("\n" + "=" * 60)
print("SCENARIO COMPARISON (ranked by cooling effectiveness)")
print("=" * 60)
print(f"\n  {'Scenario':<22} {'Target Cooling':>15} {'City-wide':>10} {'Coverage':>10}")
print(f"  {'-'*60}")
for name, r in ranked:
    print(f"  {name:<22} {r['mean_temp_reduction_target_C']:>12.2f}°C {r['overall_city_reduction_C']:>8.2f}°C {r['target_pct']:>8.1f}%")

# Save results
results_path = OUTPUT_DIR / "scenario_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved → {results_path}")

# Generate comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
names = [name.replace("_", "\n") for name, _ in ranked]
reductions = [r["mean_temp_reduction_target_C"] for _, r in ranked]
city_reductions = [r["overall_city_reduction_C"] for _, r in ranked]

x = np.arange(len(names))
bar_width = 0.35

bars1 = ax.bar(x - bar_width / 2, reductions, bar_width, label="Target Area Cooling", color="#2196F3")
bars2 = ax.bar(x + bar_width / 2, city_reductions, bar_width, label="City-wide Cooling", color="#4CAF50")

ax.set_xlabel("Intervention Scenario")
ax.set_ylabel("Temperature Reduction (°C)")
ax.set_title("UHIP - Cooling Intervention Effectiveness")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax.annotate(f"{bar.get_height():.2f}°C",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scenario_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Comparison chart saved → {OUTPUT_DIR / 'scenario_comparison.png'}")

# ─── FINAL RECOMMENDATION ───────────────────────────────────────────────────
best_name, best_result = ranked[0]
print("\n" + "=" * 60)
print("✓ OPTIMAL INTERVENTION STRATEGY")
print("=" * 60)
print(f"  Recommended: {best_name}")
print(f"  Description: {best_result['description']}")
print(f"  Expected cooling: {best_result['mean_temp_reduction_target_C']:.2f}°C in target zones")
print(f"  City-wide impact: {best_result['overall_city_reduction_C']:.2f}°C average reduction")
print(f"\n  All scenario delta maps saved as GeoTIFFs in {OUTPUT_DIR}/")
print("  Load in QGIS with Blue-White-Red colormap to visualize cooling zones.")
print("\n  Next step: python identify_hotspots.py")
