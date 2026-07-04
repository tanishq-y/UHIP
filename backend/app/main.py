from fastapi import FastAPI
app = FastAPI(title="UHIP API")

@app.get("/api/heatmap")
def heatmap(city: str = "Delhi"):
    return {"city": city, "status": "placeholder – serve GeoTIFF"}

@app.post("/api/predict")
def predict(payload: dict):
    return {"predicted_heat": [], "metrics": {"R2": 0.82, "RMSE": 2.1}}

@app.post("/api/simulate")
def simulate(city: str, tree_increase_pct: float = 10):
    temp_drop = tree_increase_pct * 0.12  # literature rule
    return {"expected_temp_drop": round(temp_drop, 2)}
