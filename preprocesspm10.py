import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("data/sample_pm_reflectance_dataset.csv")

# Drop reflectance_VIS if it exists
df = df.drop(columns=['reflectance_VIS'], errors='ignore')

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Extract temporal features
df['hour'] = df['timestamp'].dt.hour

# Drop missing pm10 values
df = df.dropna(subset=['pm10_ground'])

# Log transform PM10
df['log_pm10'] = np.log1p(df['pm10_ground'])  # log(1 + x)

# Reorder columns for clarity
columns_order = [
    'timestamp', 'hour', 'latitude', 'longitude',
    'reflectance_SWIR', 'temperature_2m', 'humidity_2m',
    'pbl_height', 'wind_speed_10m', 'pm10_ground', 'log_pm10'
]
df = df[[col for col in columns_order if col in df.columns]]

# Save processed PM10 dataset
df.to_csv("data/processed_pm10_dataset.csv", index=False)
print("✅ Saved: processed_pm10_dataset.csv")
