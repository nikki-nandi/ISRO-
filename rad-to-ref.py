import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib

# --- PAGE CONFIG ---
st.set_page_config(page_title="High-Resolution PM2.5 Map", layout="wide")

# --- TITLE ---
st.markdown("""
    <h2 style='text-align: center; color: #64b5f6;'>🌏 High-Resolution PM2.5 Prediction Map</h2>
    <h5 style='text-align: center; color: #a5b4c3;'>Using Satellite + Reanalysis Data + ML</h5>
""", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("high_res_input_sample_100.csv")
    return df

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_model_scaler():
    model = joblib.load("pm25_model.pkl")
    scaler = joblib.load("pm25_scaler.pkl")
    return model, scaler

# --- COLOR BASED ON PM2.5 LEVEL ---
def get_pm_color(pm):
    if pm <= 60:
        return [0, 200, 0]
    elif pm <= 120:
        return [255, 165, 0]
    else:
        return [255, 0, 0]

# --- MAIN ---
df = load_data()
model, scaler = load_model_scaler()

feature_cols = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
                "pbl_height", "wind_speed_10m", "hour"]

# Scale and predict
X_scaled = scaler.transform(df[feature_cols])
df["PM2.5_Predicted"] = model.predict(X_scaled)

# Add color for map
df["color"] = df["PM2.5_Predicted"].apply(get_pm_color)

# --- DISPLAY MAP ---
st.subheader("🗺️ Predicted PM2.5 Levels Across India")
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[longitude, latitude]',
    get_radius=10000,
    get_fill_color="color",
    pickable=True,
    opacity=0.8,
)

view_state = pdk.ViewState(
    latitude=22.5,
    longitude=80.0,
    zoom=4.5,
    pitch=40,
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_state,
    layers=[layer],
    tooltip={"text": "Lat: {latitude}\nLon: {longitude}\nPM2.5: {PM2.5_Predicted}"}
))

# --- TABLE & DOWNLOAD ---
with st.expander("📋 Show Prediction Table"):
    st.dataframe(df[["latitude", "longitude", "PM2.5_Predicted"]].round(2))

st.download_button(
    label="📥 Download Predictions as CSV",
    data=df.to_csv(index=False).encode(),
    file_name="pm25_high_res_predictions.csv",
    mime="text/csv"
)
