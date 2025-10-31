import pandas as pd

df = pd.read_csv("data/raw/cars.csv")
df["age"] = 2025 - df["year"]
df.to_csv("data/processed/cars_clean.csv", index=False)
print("✅ Data cleaned and saved to data/processed/")
