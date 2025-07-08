import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows for prototype dataset
num_samples = 10000

# Generate synthetic geolocations near CPCB stations
cities = {
    "Delhi": (28.61, 77.20),
    "Bangalore": (12.97, 77.59),
    "Hyderabad": (17.38, 78.48),
    "Mumbai": (19.07, 72.88),
    "Kolkata": (22.57, 88.36)
}
city_names = list(cities.keys())
city_choices = np.random.choice(city_names, num_samples)

latitudes = [cities[city][0] + np.random.uniform(-0.05, 0.05) for city in city_choices]
longitudes = [cities[city][1] + np.random.uniform(-0.05, 0.05) for city in city_choices]

# Create synthetic satellite & met features
data = {
    "city": city_choices,
    "latitude": latitudes,
    "longitude": longitudes,
    "aod": np.random.uniform(0.1, 2.5, num_samples),  # Aerosol Optical Depth
    "reflectance_SWIR": np.random.uniform(0.01, 0.5, num_samples),
    "temperature_2m": np.random.uniform(15, 45, num_samples),  # in Celsius
    "humidity_2m": np.random.uniform(20, 100, num_samples),  # in %
    "pbl_height": np.random.uniform(100, 2000, num_samples),  # in meters
    "wind_speed_10m": np.random.uniform(0, 15, num_samples),  # in m/s
    "hour": np.random.randint(0, 24, num_samples),
}

df = pd.DataFrame(data)

# Generate synthetic PM2.5 and PM10 values using noisy combinations
df["PM2.5"] = (
    df["aod"] * 40 +
    df["humidity_2m"] * 0.1 +
    df["temperature_2m"] * 0.2 -
    df["wind_speed_10m"] * 1.5 +
    np.random.normal(0, 10, num_samples)
).clip(5, 400)

df["PM10"] = (
    df["aod"] * 60 +
    df["humidity_2m"] * 0.15 +
    df["temperature_2m"] * 0.3 +
    df["pbl_height"] * 0.01 -
    df["wind_speed_10m"] * 1.2 +
    np.random.normal(0, 15, num_samples)
).clip(10, 500)

# Reorder columns
df = df[[
    "city", "latitude", "longitude", "aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
    "pbl_height", "wind_speed_10m", "hour", "PM2.5", "PM10"
]]

# Save to CSV
sample_path = "data/realworld_prototype_air_quality.csv"
df.to_csv(sample_path, index=False)

sample_path
