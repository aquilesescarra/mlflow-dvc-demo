import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_experiment("cars_price_pred")

df = pd.read_csv("data/processed/cars_clean.csv")
X = df[["mileage", "age"]]
y = df["price"]

model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)
preds = model.predict(X)
mae = mean_absolute_error(y, preds)

mlflow.log_param("n_estimators", 10)
mlflow.log_metric("mae", mae)
mlflow.sklearn.log_model(model, "model")

joblib.dump(model, "models/rf_model.pkl")
print(f"✅ Model trained! MAE={mae:.2f}")
