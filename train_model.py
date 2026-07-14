"""
train_model.py
UHIP - AI/ML Model for Urban Heat Dynamics
Trains XGBoost regressor to predict LST from urban drivers,
performs SHAP analysis for driver quantification, and saves model for inference.

Run: python train_model.py
"""
import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_FILES = {
    "ndvi": PROC / "NDVI_COG.tif",
    "ndbi": PROC / "NDBI_COG.tif",
    "build_density": PROC / "BUILD_DENSITY_COG.tif",
}
TARGET_FILE = PROC / "LST_Celsius_COG.tif"

# Fallback to non-COG if COG doesn't exist
for key, path in FEATURE_FILES.items():
    if not path.exists():
        FEATURE_FILES[key] = PROC / path.name.replace("_COG", "")

if not TARGET_FILE.exists():
    TARGET_FILE = PROC / "LST_Celsius.tif"

SAMPLE_FRACTION = 0.1  # use 10% of valid pixels for training (memory-friendly)
RANDOM_STATE = 42


# ─── 1. LOAD RASTERS ────────────────────────────────────────────────────────
def load_raster(path):
    """Load single-band raster as float32 array."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, src.profile


print("=" * 60)
print("UHIP - Training Urban Heat Dynamics Model")
print("=" * 60)

print("\n[1/7] Loading rasters...")
target, target_profile = load_raster(TARGET_FILE)
print(f"  LST shape: {target.shape}, range: {np.nanmin(target):.1f} - {np.nanmax(target):.1f} °C")

features = {}
for name, path in FEATURE_FILES.items():
    arr, _ = load_raster(path)
    features[name] = arr
    print(f"  {name} shape: {arr.shape}, range: {np.nanmin(arr):.3f} - {np.nanmax(arr):.3f}")


# ─── 2. ENGINEER ADDITIONAL FEATURES ────────────────────────────────────────
print("\n[2/7] Engineering additional features...")

# NDVI squared (captures non-linear cooling effect of vegetation)
features["ndvi_sq"] = features["ndvi"] ** 2

# NDBI × build_density interaction (amplifies heat in built-up dense areas)
features["ndbi_x_build"] = features["ndbi"] * features["build_density"]

# Vegetation deficit: how far below max greenness
features["veg_deficit"] = 1.0 - features["ndvi"]

print(f"  Total features: {len(features)}")
for name in features:
    print(f"    - {name}")


# ─── 3. BUILD TABULAR DATASET ────────────────────────────────────────────────
print("\n[3/7] Building tabular dataset...")

# Flatten all arrays
h, w = target.shape
y_flat = target.ravel()

X_dict = {}
for name, arr in features.items():
    X_dict[name] = arr.ravel()

# Create valid mask (no NaN in any layer)
valid_mask = np.isfinite(y_flat)
for arr_flat in X_dict.values():
    valid_mask &= np.isfinite(arr_flat)

# Also exclude extreme outliers in LST
valid_mask &= (y_flat > 15) & (y_flat < 65)

n_valid = valid_mask.sum()
print(f"  Total pixels: {h * w:,}")
print(f"  Valid pixels: {n_valid:,} ({100 * n_valid / (h * w):.1f}%)")

# Subsample for training efficiency
np.random.seed(RANDOM_STATE)
valid_indices = np.where(valid_mask)[0]
n_sample = int(n_valid * SAMPLE_FRACTION)
sample_indices = np.random.choice(valid_indices, size=n_sample, replace=False)

feature_names = list(X_dict.keys())
X = np.column_stack([X_dict[name][sample_indices] for name in feature_names])
y = y_flat[sample_indices]

print(f"  Sampled for training: {n_sample:,} pixels")
print(f"  Feature matrix shape: {X.shape}")


# ─── 4. TRAIN/TEST SPLIT & MODEL TRAINING ───────────────────────────────────
print("\n[4/7] Training XGBoost model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)


# ─── 5. EVALUATE ─────────────────────────────────────────────────────────────
print("\n[5/7] Evaluating model...")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

metrics = {
    "train": {
        "R2": round(float(r2_score(y_train, y_pred_train)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_train, y_pred_train))), 4),
        "MAE": round(float(mean_absolute_error(y_train, y_pred_train)), 4),
    },
    "test": {
        "R2": round(float(r2_score(y_test, y_pred_test)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 4),
        "MAE": round(float(mean_absolute_error(y_test, y_pred_test)), 4),
    },
}

print(f"\n  {'Metric':<8} {'Train':>10} {'Test':>10}")
print(f"  {'-'*30}")
for metric in ["R2", "RMSE", "MAE"]:
    print(f"  {metric:<8} {metrics['train'][metric]:>10.4f} {metrics['test'][metric]:>10.4f}")


# ─── 6. SHAP ANALYSIS (Driver Quantification) ───────────────────────────────
print("\n[6/7] Computing SHAP values for driver analysis...")

# Use a subset for SHAP (it's computationally expensive)
shap_sample_size = min(5000, len(X_test))
X_shap = X_test[:shap_sample_size]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# Mean absolute SHAP values = feature importance
mean_shap = np.abs(shap_values).mean(axis=0)
driver_importance = {
    feature_names[i]: round(float(mean_shap[i]), 4)
    for i in range(len(feature_names))
}

# Sort by importance
driver_importance = dict(sorted(driver_importance.items(), key=lambda x: x[1], reverse=True))

print("\n  Driver Contribution to Urban Heating (mean |SHAP|):")
print(f"  {'Driver':<20} {'Importance':>12}")
print(f"  {'-'*34}")
for driver, importance in driver_importance.items():
    bar = "█" * int(importance / max(driver_importance.values()) * 20)
    print(f"  {driver:<20} {importance:>10.4f}  {bar}")

# Save SHAP summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  SHAP summary plot saved → {MODEL_DIR / 'shap_summary.png'}")

# Save SHAP bar plot
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(MODEL_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  SHAP bar plot saved → {MODEL_DIR / 'shap_bar.png'}")


# ─── 7. SAVE MODEL & METADATA ───────────────────────────────────────────────
print("\n[7/7] Saving model and metadata...")

# Save model
model_path = MODEL_DIR / "xgb_lst_model.json"
model.save_model(str(model_path))
print(f"  Model saved → {model_path}")

# Also save as joblib for sklearn compatibility
joblib_path = MODEL_DIR / "xgb_lst_model.joblib"
joblib.dump(model, joblib_path)
print(f"  Joblib saved → {joblib_path}")

# Save metadata
metadata = {
    "project": "UHIP",
    "model_type": "XGBRegressor",
    "target": "LST_Celsius",
    "features": feature_names,
    "feature_importance_shap": driver_importance,
    "metrics": metrics,
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "hyperparameters": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "city": "Delhi",
    "crs": "EPSG:32643",
}

meta_path = MODEL_DIR / "model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Metadata saved → {meta_path}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✓ MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"  Test R²:   {metrics['test']['R2']}")
print(f"  Test RMSE: {metrics['test']['RMSE']} °C")
print(f"  Test MAE:  {metrics['test']['MAE']} °C")
print(f"\n  Top drivers of urban heating:")
for i, (driver, imp) in enumerate(driver_importance.items()):
    if i >= 3:
        break
    print(f"    {i+1}. {driver} (SHAP importance: {imp})")
print(f"\n  Files saved in: {MODEL_DIR}/")
print("  Next step: python simulate_cooling.py")
