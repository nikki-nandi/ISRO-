import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import os
import gdown
import altair as alt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="PM2.5 & PM10 Monitoring", layout="wide")

# ------------------ STYLE ------------------
st.markdown("""
    <style>
    .main { background-color: #0b1725; color: #ffffff; }
    section[data-testid="stSidebar"] {
        background-color: #08121d;
        color: white;
        border-right: 1px solid #222;
    }
    h1, h2, h3, h4, .st-bb, .st-cb { color: #ffffff !important; }
    .stButton>button, .stDownloadButton>button {
        background-color: #1464b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: bold;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        color: black !important;
        background-color: white !important;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ DOWNLOAD MODELS ------------------
@st.cache_resource
def download_models():
    os.makedirs("models", exist_ok=True)
    model_files = {
        "models/pm25_model.pkl": "1WGNp0FsvcLtSIbfk2ZSvOu7QD6nrjZea",
        "models/pm10_model.pkl": "169669rOcO1zcfiyoZqVW_XLj6YtO5xk3",
        "models/pm25_scaler.pkl": "134ahvy25P4yTlXLdt5DdaHyUerL1IUv7",
        "models/pm10_scaler.pkl": "1rTZb-CgQpkrrnOkE43UiXtFtFKQDPjde"
    }
    for path, fid in model_files.items():
        if not os.path.exists(path):
            gdown.download(f"https://drive.google.com/uc?id={fid}", path, quiet=False)

download_models()

# ------------------ LOAD MODELS ------------------
pm25_model = joblib.load("models/pm25_model.pkl")
pm10_model = joblib.load("models/pm10_model.pkl")
pm25_scaler = joblib.load("models/pm25_scaler.pkl")
pm10_scaler = joblib.load("models/pm10_scaler.pkl")

# ------------------ HEADER ------------------
col1, col2, col3 = st.columns([1, 5, 1])
with col1:
    st.image("ISRO-Color.png", width=150)
with col2:
    st.markdown("<h2 style='text-align:center;color:#64b5f6;'>ISRO & CPCB AIR POLLUTION MONITOR</h2><h5 style='text-align:center;color:#a5b4c3;'>Live Air Quality Monitoring Dashboard</h5>", unsafe_allow_html=True)
with col3:
    st.image("cpcb.png", width=150)

st.markdown("---")

# ------------------ MAP COLOR FUNCTION ------------------
def get_pm_color(pm):
    if pm <= 60:
        return [0, 200, 0]
    elif pm <= 120:
        return [255, 165, 0]
    else:
        return [255, 0, 0]

# ------------------ HIGH-RES PREDICTION MAP ------------------
st.subheader("🌍 High-Resolution PM2.5 Prediction Map")
@st.cache_data
def load_high_res_data():
    return pd.read_csv("data/high_res_input_sample_100.csv")

df_map = load_high_res_data()
features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m", "pbl_height", "wind_speed_10m", "hour"]
X_scaled = pm25_scaler.transform(df_map[features])
df_map["PM2.5_Predicted"] = pm25_model.predict(X_scaled)
df_map["color"] = df_map["PM2.5_Predicted"].apply(get_pm_color)

layer_map = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[longitude, latitude]',
    get_radius=10000,
    get_fill_color="color",
    pickable=True,
    opacity=0.8
)

view_map = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=4.5, pitch=40)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_map,
    layers=[layer_map],
    tooltip={"text": "Lat: {latitude}\nLon: {longitude}\nPM2.5: {PM2.5_Predicted:.2f}"}
))

# ------------------ CITY MONITORING ------------------
st.subheader("📊 Multi-City PM2.5 & PM10 Predictions")

available_cities = {
    "Delhi": "data/delhi_pm_data.csv",
    "Bangalore": "data/bangalore_pm_data.csv",
    "Hyderabad": "data/hyderabad_pm_data.csv",
    "Kolkata": "data/kolkata_pm_data.csv"
}

st.sidebar.header("🔧 Configuration")
selected_cities = st.sidebar.multiselect("Select cities to monitor:", list(available_cities.keys()), default=["Delhi"])
limit_rows = st.sidebar.slider("Recent rows to visualize (per city):", min_value=1, max_value=20, value=10)

df_all = []
for city in selected_cities:
    path = available_cities.get(city)
    if os.path.exists(path):
        df = pd.read_csv(path).tail(limit_rows)
        df["city"] = city
        df_all.append(df)

if not df_all:
    st.warning("No data available.")
    st.stop()

df_all = pd.concat(df_all, ignore_index=True)

df_all["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(df_all[features]))
df_all["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(df_all[features]))

for city in selected_cities:
    st.markdown(f"### 🏙️ {city}")
    df_city = df_all[df_all["city"] == city]

    col1, col2 = st.columns(2)
    col1.metric("Latest PM2.5", f"{df_city['PM2.5_pred'].iloc[-1]:.2f} µg/m³")
    col2.metric("Latest PM10", f"{df_city['PM10_pred'].iloc[-1]:.2f} µg/m³")

    view = pdk.ViewState(latitude=df_city["latitude"].iloc[-1], longitude=df_city["longitude"].iloc[-1], zoom=6, pitch=30)
    layer = pdk.Layer("ScatterplotLayer", data=df_city, get_position='[longitude, latitude]', get_fill_color=df_city["PM2.5_pred"].apply(get_pm_color), get_radius=10000)
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/dark-v10", initial_view_state=view, layers=[layer]))

    melted = pd.melt(df_city, id_vars=["hour"], value_vars=["PM2.5_pred", "PM10_pred"], var_name="Pollutant", value_name="Concentration")
    chart = alt.Chart(melted).mark_line(point=True).encode(
        x=alt.X("hour:O", title="Hour"),
        y=alt.Y("Concentration:Q", title="µg/m³"),
        color="Pollutant:N"
    ).properties(height=350, width=800)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("---")

# ------------------ DOWNLOAD BUTTON ------------------
st.download_button(
    label="⬇️ Download Predictions",
    data=df_all.to_csv(index=False).encode(),
    file_name="citywise_predictions.csv",
    mime="text/csv"
)
