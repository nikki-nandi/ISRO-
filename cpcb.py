import requests
import pandas as pd

# === Step 1: Define valid OpenAQ location IDs ===
# Use IDs that exist in OpenAQ's current archive for India
# These are just examples — you can update as needed
locations = {
    "Delhi_AnandVihar": 147527,       # Replace with working IDs
    "Hyderabad_ZooPark": 147962,
    "Bengaluru_SilkBoard": 149199
}

date = "2024-01-23"
parameters = ["pm25", "pm10"]
records = []

# === Step 2: Fetch data for each location and parameter ===
for name, loc_id in locations.items():
    for param in parameters:
        print(f"📡 Fetching {param.upper()} for {name}")
        url = "https://api.openaq.org/v2/measurements"
        params = {
            "location_id": loc_id,
            "parameter": param,
            "date_from": f"{date}T00:00:00Z",
            "date_to": f"{date}T23:59:59Z",
            "limit": 100,
            "sort": "desc"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            for r in data.get("results", []):
                records.append({
                    "Location": name,
                    "Latitude": r["coordinates"]["latitude"],
                    "Longitude": r["coordinates"]["longitude"],
                    "Datetime": r["date"]["utc"],
                    "Parameter": r["parameter"].upper(),  # PM2.5 or PM10
                    "Value": r["value"]
                })

        except Exception as e:
            print(f"❌ Error for {name} - {param}: {e}")

# === Step 3: Convert to DataFrame ===
df = pd.DataFrame(records)

if df.empty:
    print("⚠️ No data found. Check location IDs or date.")
else:
    # === Step 4: Pivot to wide format ===
    df_wide = df.pivot_table(
        index=["Location", "Latitude", "Longitude", "Datetime"],
        columns="Parameter",
        values="Value"
    ).reset_index()

    # === Optional: Rename columns ===
    df_wide.rename(columns={"PM2.5": "PM2.5", "PM10": "PM10"}, inplace=True)

    # === Step 5: Save to CSV ===
    df_wide.to_csv("cpcb_pm_data.csv", index=False)
    print("✅ PM2.5 and PM10 data saved to 'cpcb_pm25_pm10_openaq.csv'")
