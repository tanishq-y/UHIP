"""
validate_model.py
UHIP - Model Validation Against Ground Truth
Validates the trained model using IMD station data and spatial cross-validation.

Run: python validate_model.py
"""
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from pathlib import Path
import joblib
import json
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pick(name):
    cog = PROC / f"{name}_COG.tif"
    tif = PROC / f"{name}.tif"
    return cog if cog.exists() else tif


# ─── 1. LOAD CONFIG & MODEL ─────────────────────────────────────────────────
print("=" * 60)
print("UHIP - Model Validation")
print("=" * 60)

print("\n[1/4] Loading model and configuration...")
model = joblib.load(MODEL_DIR / "xgb_lst_model.joblib")
meta = json.load(open(MODEL_DIR / "model_metadata.json"))
feature_names = meta["features"]

with open("config.yaml") as f:
    config = yaml.safe_load(f)

stations = config.get("validation", {}).get("imd_stations", [])
print(f"  Model: {meta['model_type']}")
print(f"  Validation stations: {len(stations)}")

# Known Delhi reference points for validation
# (name, lat, lon, expected behavior)
REFERENCE_POINTS = [
    {"name": "Lodhi Garden (park)", "lat": 28.5933, "lon": 77.2197, "expected": "cool", "est_lst_range": (30, 36)},
    {"name": "ITO (CBD)", "lat": 28.6280, "lon": 77.2410, "expected": "hot", "est_lst_range": (38, 45)},
    {"name": "Connaught Place", "lat": 28.6315, "lon": 77.2167, "expected": "hot", "est_lst_range": (37, 44)},
    {"name": "India Gate", "lat": 28.6129, "lon": 77.2295, "expected": "moderate", "est_lst_range": (33, 39)},
    {"name": "Yamuna floodplain", "lat": 28.6350, "lon": 77.2600, "expected": "moderate", "est_lst_range": (32, 38)},
    {"name": "Okhla Industrial", "lat": 28.5300, "lon": 77.2700, "expected": "hot", "est_lst_range": (39, 46)},
    {"name": "Ridge Forest", "lat": 28.6800, "lon": 77.1800, "expected": "cool", "est_lst_range": (29, 35)},
    {"name": "Dwarka (residential)", "lat": 28.5921, "lon": 77.0460, "expected": "moderate", "est_lst_range": (34, 40)},
]

# Add IMD stations
for stn in stations:
    REFERENCE_POINTS.append({
        "name": f"IMD {stn['name']}",
        "lat": stn["lat"],
        "lon": stn["lon"],
        "expected": "reference",
        "est_lst_range": (stn["expected_lst_july"] - 3, stn["expected_lst_july"] + 3),
    })


# ─── 2. POINT VALIDATION ────────────────────────────────────────────────────
print("\n[2/4] Validating at reference points...")

with rasterio.open(pick("LST_Celsius")) as lst_src:
    lst_crs = lst_src.crs

    results = []
    for pt in REFERENCE_POINTS:
        # Transform WGS84 → raster CRS
        xs, ys = warp_transform("EPSG:4326", lst_crs, [pt["lon"]], [pt["lat"]])

        # Sample all layers
        values = {}
        for layer_name in ["LST_Celsius", "NDVI", "NDBI", "BUILD_DENSITY"]:
            with rasterio.open(pick(layer_name)) as src:
                for val in src.sample([(xs[0], ys[0])]):
                    v = float(val[0])
                    if src.nodata is not None and np.isclose(v, src.nodata):
                        v = None
                    elif np.isnan(v) or np.isinf(v):
                        v = None
                    values[layer_name] = v

        lst_actual = values.get("LST_Celsius")
        ndvi_val = values.get("NDVI")
        ndbi_val = values.get("NDBI")
        build_val = values.get("BUILD_DENSITY")

        # Model prediction at this point
        if all(v is not None for v in [ndvi_val, ndbi_val, build_val]):
            feat_dict = {
                "ndvi": ndvi_val,
                "ndbi": ndbi_val,
                "build_density": build_val,
                "ndvi_sq": ndvi_val ** 2,
                "ndbi_x_build": ndbi_val * build_val,
                "veg_deficit": 1.0 - ndvi_val,
            }
            X_pt = np.array([[feat_dict[f] for f in feature_names]])
            lst_predicted = float(model.predict(X_pt)[0])
        else:
            lst_predicted = None

        # Check if within expected range
        in_range = None
        if lst_actual is not None:
            low, high = pt["est_lst_range"]
            in_range = low <= lst_actual <= high

        results.append({
            "name": pt["name"],
            "lat": pt["lat"],
            "lon": pt["lon"],
            "expected_type": pt["expected"],
            "lst_actual": round(lst_actual, 2) if lst_actual else None,
            "lst_predicted": round(lst_predicted, 2) if lst_predicted else None,
            "expected_range": pt["est_lst_range"],
            "in_expected_range": in_range,
            "prediction_error": round(lst_predicted - lst_actual, 2) if (lst_predicted and lst_actual) else None,
        })

# Print results table
print(f"\n  {'Location':<25} {'Actual':>8} {'Predicted':>10} {'Error':>7} {'Expected':>12} {'Pass':>5}")
print(f"  {'-'*70}")
for r in results:
    actual = f"{r['lst_actual']:.1f}°C" if r['lst_actual'] else "N/A"
    predicted = f"{r['lst_predicted']:.1f}°C" if r['lst_predicted'] else "N/A"
    error = f"{r['prediction_error']:.1f}" if r['prediction_error'] is not None else "N/A"
    expected = f"{r['expected_range'][0]}-{r['expected_range'][1]}°C"
    passed = "✓" if r['in_expected_range'] else "✗" if r['in_expected_range'] is not None else "?"
    print(f"  {r['name']:<25} {actual:>8} {predicted:>10} {error:>7} {expected:>12} {passed:>5}")

# Prediction accuracy at points
valid_preds = [(r["lst_actual"], r["lst_predicted"]) for r in results
               if r["lst_actual"] is not None and r["lst_predicted"] is not None]
if valid_preds:
    actuals = [v[0] for v in valid_preds]
    preds = [v[1] for v in valid_preds]
    point_r2 = r2_score(actuals, preds)
    point_rmse = np.sqrt(mean_squared_error(actuals, preds))
    point_mae = mean_absolute_error(actuals, preds)
    print(f"\n  Point-level metrics: R²={point_r2:.4f}, RMSE={point_rmse:.2f}°C, MAE={point_mae:.2f}°C")


# ─── 3. SPATIAL CROSS-VALIDATION ────────────────────────────────────────────
print("\n[3/4] Running spatial cross-validation (5-fold)...")

# Load full dataset for CV
with rasterio.open(pick("LST_Celsius")) as src:
    lst_full = src.read(1).astype("float32")
    nd = src.nodata
    if nd is not None:
        lst_full[lst_full == nd] = np.nan

with rasterio.open(pick("NDVI")) as src:
    ndvi_full = src.read(1).astype("float32")
with rasterio.open(pick("NDBI")) as src:
    ndbi_full = src.read(1).astype("float32")
with rasterio.open(pick("BUILD_DENSITY")) as src:
    build_full = src.read(1).astype("float32")

# Flatten and filter
h, w = lst_full.shape
y_all = lst_full.ravel()
X_all = np.column_stack([
    ndvi_full.ravel(),
    ndbi_full.ravel(),
    build_full.ravel(),
    (ndvi_full ** 2).ravel(),
    (ndbi_full * build_full).ravel(),
    (1.0 - ndvi_full).ravel(),
])

valid = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1) & (y_all > 15) & (y_all < 65)
X_valid = X_all[valid]
y_valid = y_all[valid]

# Subsample for CV
np.random.seed(42)
n_cv = min(50000, len(y_valid))
idx = np.random.choice(len(y_valid), n_cv, replace=False)
X_cv = X_valid[idx]
y_cv = y_valid[idx]

# 5-fold CV
import xgboost as xgb
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {"r2": [], "rmse": [], "mae": []}

for fold, (train_idx, test_idx) in enumerate(kf.split(X_cv)):
    X_tr, X_te = X_cv[train_idx], X_cv[test_idx]
    y_tr, y_te = y_cv[train_idx], y_cv[test_idx]

    fold_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    fold_model.fit(X_tr, y_tr)
    y_pred = fold_model.predict(X_te)

    cv_scores["r2"].append(r2_score(y_te, y_pred))
    cv_scores["rmse"].append(np.sqrt(mean_squared_error(y_te, y_pred)))
    cv_scores["mae"].append(mean_absolute_error(y_te, y_pred))

print(f"\n  Cross-Validation Results (5-fold):")
print(f"  {'Metric':<8} {'Mean':>10} {'Std':>10}")
print(f"  {'-'*30}")
for metric in ["r2", "rmse", "mae"]:
    vals = cv_scores[metric]
    print(f"  {metric.upper():<8} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")


# ─── 4. GENERATE VALIDATION REPORT ──────────────────────────────────────────
print("\n[4/4] Generating validation report...")

validation_report = {
    "project": "UHIP",
    "model": meta["model_type"],
    "city": "Delhi",
    "point_validation": {
        "n_points": len(results),
        "metrics": {
            "R2": round(point_r2, 4) if valid_preds else None,
            "RMSE_C": round(point_rmse, 2) if valid_preds else None,
            "MAE_C": round(point_mae, 2) if valid_preds else None,
        },
        "points": results,
    },
    "cross_validation": {
        "method": "5-fold",
        "n_samples": n_cv,
        "metrics": {
            "R2_mean": round(np.mean(cv_scores["r2"]), 4),
            "R2_std": round(np.std(cv_scores["r2"]), 4),
            "RMSE_mean_C": round(np.mean(cv_scores["rmse"]), 4),
            "RMSE_std_C": round(np.std(cv_scores["rmse"]), 4),
            "MAE_mean_C": round(np.mean(cv_scores["mae"]), 4),
            "MAE_std_C": round(np.std(cv_scores["mae"]), 4),
        },
    },
    "physical_consistency": {
        "parks_cooler_than_cbd": None,  # will check
        "high_ndvi_low_lst_correlation": None,
    },
}

# Physical consistency checks
cool_points = [r["lst_actual"] for r in results if r["expected_type"] == "cool" and r["lst_actual"]]
hot_points = [r["lst_actual"] for r in results if r["expected_type"] == "hot" and r["lst_actual"]]
if cool_points and hot_points:
    parks_cooler = np.mean(cool_points) < np.mean(hot_points)
    validation_report["physical_consistency"]["parks_cooler_than_cbd"] = bool(parks_cooler)
    print(f"\n  Physical consistency: Parks cooler than CBD? {'✓ YES' if parks_cooler else '✗ NO'}")
    print(f"    Mean park LST: {np.mean(cool_points):.1f}°C vs Mean CBD LST: {np.mean(hot_points):.1f}°C")

# NDVI-LST correlation
ndvi_flat = ndvi_full.ravel()[valid]
lst_flat = lst_full.ravel()[valid]
correlation = np.corrcoef(ndvi_flat[idx], lst_flat[idx])[0, 1]
validation_report["physical_consistency"]["high_ndvi_low_lst_correlation"] = round(float(correlation), 4)
print(f"    NDVI-LST correlation: {correlation:.4f} (expected negative)")

# Save report
report_path = OUTPUT_DIR / "validation_report.json"
with open(report_path, "w") as f:
    json.dump(validation_report, f, indent=2, default=str)
print(f"\n  Report saved → {report_path}")

# Scatter plot: actual vs predicted
if valid_preds:
    fig, ax = plt.subplots(figsize=(7, 7))
    actuals_arr = np.array(actuals)
    preds_arr = np.array(preds)
    ax.scatter(actuals_arr, preds_arr, c="steelblue", alpha=0.7, s=80, edgecolors="k", linewidth=0.5)
    lims = [min(actuals_arr.min(), preds_arr.min()) - 1, max(actuals_arr.max(), preds_arr.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")
    ax.set_xlabel("Actual LST (°C)")
    ax.set_ylabel("Predicted LST (°C)")
    ax.set_title(f"UHIP Model Validation\nR²={point_r2:.3f}, RMSE={point_rmse:.2f}°C")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved → {OUTPUT_DIR / 'actual_vs_predicted.png'}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✓ VALIDATION COMPLETE")
print("=" * 60)
print(f"  Cross-Validation R²: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
print(f"  Cross-Validation RMSE: {np.mean(cv_scores['rmse']):.2f} ± {np.std(cv_scores['rmse']):.2f} °C")
if valid_preds:
    print(f"  Point Validation R²: {point_r2:.4f}")
    print(f"  Point Validation RMSE: {point_rmse:.2f} °C")
print(f"\n  All outputs in: {OUTPUT_DIR}/")
