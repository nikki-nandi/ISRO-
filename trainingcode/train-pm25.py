import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Step 1: Load preprocessed dataset
df = pd.read_csv("processed_pm_dataset.csv")  # Replace with your actual file

# Step 2: Define features and target
features = [
    'reflectance_SWIR',
    'temperature_2m',
    'humidity_2m',
    'pbl_height',
    'wind_speed_10m',
    'hour'
]
target = 'log_pm25'  # Using log-transformed PM2.5 values

# Step 3: Split dataset
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Step 6: Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n📊 Cross-Validation R² Scores: {cv_scores}")
print(f"📊 Mean CV R²: {np.mean(cv_scores):.2f}")

# Step 7: Predict and inverse log
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Step 8: Evaluate
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\n✅ Model trained for PM2.5")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Step 9: Save model and scaler
joblib.dump(model, "pm25_model.pkl")
joblib.dump(scaler, "pm25_scaler.pkl")
print("💾 Saved: pm25_model.pkl and pm25_scaler.pkl")

# Step 10: Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5")
plt.grid(True)
plt.tight_layout()
plt.show()
