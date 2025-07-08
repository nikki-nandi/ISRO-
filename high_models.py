import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Generate fake input data
df = pd.DataFrame({
    "aod": np.random.uniform(0.1, 1.2, 1000),
    "reflectance_SWIR": np.random.uniform(0.01, 0.3, 1000),
    "temperature_2m": np.random.uniform(15, 45, 1000),
    "humidity_2m": np.random.uniform(20, 100, 1000),
    "pbl_height": np.random.uniform(100, 3000, 1000),
    "wind_speed_10m": np.random.uniform(0.5, 10, 1000),
    "hour": np.random.randint(0, 24, 1000)
})

df["PM2.5"] = df["aod"] * 80 + df["humidity_2m"] * 0.2 + df["temperature_2m"] * 0.5
df["PM10"] = df["aod"] * 120 + df["humidity_2m"] * 0.25 + df["temperature_2m"] * 0.6

features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m", "pbl_height", "wind_speed_10m", "hour"]

# PM2.5
scaler25 = StandardScaler()
X25 = scaler25.fit_transform(df[features])
model25 = RandomForestRegressor().fit(X25, df["PM2.5"])
joblib.dump(model25, "pm25_model_high.pkl")
joblib.dump(scaler25, "pm25_scaler_high.pkl")

# PM10
scaler10 = StandardScaler()
X10 = scaler10.fit_transform(df[features])
model10 = RandomForestRegressor().fit(X10, df["PM10"])
joblib.dump(model10, "pm10_model_high.pkl")
joblib.dump(scaler10, "pm10_scaler_high.pkl")
