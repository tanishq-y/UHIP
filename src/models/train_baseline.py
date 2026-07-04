'''
Step 3 – Train baseline XGBoost on pixel features
Expects a CSV with columns: ndvi, ndbi, lst, temp_2m, humidity, building_density, pop_density, ...
'''
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

df = pd.read_csv("data/processed/features.csv")  # you will create this
X = df.drop(columns=["lst"])
y = df["lst"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("R2:", r2_score(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))

joblib.dump(model, "src/models/xgb_baseline.pkl")
