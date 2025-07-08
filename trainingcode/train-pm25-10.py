# Step-by-step Python code for training an AI/ML model to estimate PM2.5 and PM10
# from INSAT satellite data (e.g., reflectance/AOD), reanalysis data, and CPCB measurements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the combined dataset (INSAT + Reanalysis + CPCB)
df = pd.read_csv("realworld_prototype_air_quality.csv")

# 2. Define features and targets
features = [
    "aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
    "pbl_height", "wind_speed_10m", "hour"
]
target_pm25 = "PM2.5"
target_pm10 = "PM10"

X = df[features]
y_pm25 = df[target_pm25]
y_pm10 = df[target_pm10]

# 3. Split into training and testing sets
X_train, X_test, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
_, _, y_pm10_train, y_pm10_test = train_test_split(X, y_pm10, test_size=0.2, random_state=42)

# 4. Standardize the features
scaler_pm25 = StandardScaler()
X_train_scaled_pm25 = scaler_pm25.fit_transform(X_train)
X_test_scaled_pm25 = scaler_pm25.transform(X_test)

scaler_pm10 = StandardScaler()
X_train_scaled_pm10 = scaler_pm10.fit_transform(X_train)
X_test_scaled_pm10 = scaler_pm10.transform(X_test)

# 5. Train Random Forest models
model_pm25 = RandomForestRegressor(n_estimators=100, random_state=42)
model_pm25.fit(X_train_scaled_pm25, y_pm25_train)

model_pm10 = RandomForestRegressor(n_estimators=100, random_state=42)
model_pm10.fit(X_train_scaled_pm10, y_pm10_train)

# 6. Evaluate the models
pm25_preds = model_pm25.predict(X_test_scaled_pm25)
pm10_preds = model_pm10.predict(X_test_scaled_pm10)

print("PM2.5 Evaluation:")
print("MSE:", mean_squared_error(y_pm25_test, pm25_preds))
print("R2:", r2_score(y_pm25_test, pm25_preds))
print("Accuracy (PM2.5):", model_pm25.score(X_test_scaled_pm25, y_pm25_test))

print("\nPM10 Evaluation:")
print("MSE:", mean_squared_error(y_pm10_test, pm10_preds))
print("R2:", r2_score(y_pm10_test, pm10_preds))
print("Accuracy (PM10):", model_pm10.score(X_test_scaled_pm10, y_pm10_test))

# 7. Plot Actual vs Predicted for PM2.5
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pm25_test, y=pm25_preds, alpha=0.5, color='blue')
plt.plot([y_pm25_test.min(), y_pm25_test.max()], [y_pm25_test.min(), y_pm25_test.max()], 'r--')
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Plot Actual vs Predicted for PM10
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pm10_test, y=pm10_preds, alpha=0.5, color='green')
plt.plot([y_pm10_test.min(), y_pm10_test.max()], [y_pm10_test.min(), y_pm10_test.max()], 'r--')
plt.xlabel("Actual PM10")
plt.ylabel("Predicted PM10")
plt.title("Actual vs Predicted PM10")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Save the trained models and scalers
joblib.dump(model_pm25, "pm25_model.pkl")
joblib.dump(scaler_pm25, "pm25_scaler.pkl")

joblib.dump(model_pm10, "pm10_model.pkl")
joblib.dump(scaler_pm10, "pm10_scaler.pkl")

print("\n✅ Models and scalers saved.")
