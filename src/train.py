import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import mlflow
import mlflow.sklearn

# Crear carpeta de modelos si no existe
os.makedirs("models", exist_ok=True)

# Configurar experimento
mlflow.set_experiment("cars_price_pred")

# Cargar datos
df = pd.read_csv("data/processed/cars_clean.csv")
X = df[["mileage", "age"]].astype(float)
y = df["price"]

# Entrenar modelo
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)
preds = model.predict(X)
mae = mean_absolute_error(y, preds)

# Log en MLflow con ejemplo de entrada
input_example = X.head(1)
mlflow.log_param("n_estimators", 10)
mlflow.log_metric("mae", mae)
mlflow.sklearn.log_model(
    model,
    name="model",
    input_example=input_example
)


joblib.dump(model, "models/rf_model.pkl")
print(f"✅ Model trained! MAE={mae:.2f}")