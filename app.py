import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import time
import os
import smtplib
import altair as alt
import gdown
from prophet import Prophet
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta

# --- MODEL DOWNLOAD ---
def download_models():
    os.makedirs("models", exist_ok=True)

    model_files = {
        "models/pm25_model.pkl": "1WGNp0FsvcLtSIbfk2ZSvOu7QD6nrjZea",
        "models/pm10_model.pkl": "169669rOcO1zcfiyoZqVW_XLj6YtO5xk3",
        "models/pm25_scaler.pkl": "134ahvy25P4yTlXLdt5DdaHyUerL1IUv7",
        "models/pm10_scaler.pkl": "1rTZb-CgQpkrrnOkE43UiXtFtFKQDPjde"
    }

    for path, file_id in model_files.items():
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

download_models()

# --- LOAD MODELS ---
pm25_model = joblib.load("models/pm25_model.pkl")
pm10_model = joblib.load("models/pm10_model.pkl")
pm25_scaler = joblib.load("models/pm25_scaler.pkl")
pm10_scaler = joblib.load("models/pm10_scaler.pkl")

# --- CONFIGURATION ---
EMAIL_SENDER = "nikithnandi08@gmail.com"
EMAIL_PASSWORD = "sshz jpyi pibg jxev"
EMAIL_RECEIVER = "nikithnandi08@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
PM25_THRESHOLD = 120
PM10_THRESHOLD = 250

email_recipients = {
    "Delhi": EMAIL_RECEIVER,
    "Bangalore": EMAIL_RECEIVER,
    "Hyderabad": EMAIL_RECEIVER,
    "Kolkata": EMAIL_RECEIVER
}

# --- PAGE CONFIG ---
st.set_page_config(page_title="PM2.5 & PM10 Monitoring Dashboard", layout="wide")

# --- DARK MODE CSS + Sidebar Label Styling ---
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

# --- HEADER ---
col_logo1, col_title, col_logo2 = st.columns([1, 5, 1])
with col_logo1:
    st.image("ISRO-Color.png", width=150)
with col_title:
    st.markdown("""
        <h2 style='text-align: center; color: #64b5f6;'>ISRO & CPCB AIR POLLUTION LIVE MONITORING SITE</h2>
        <h5 style='text-align: center; color: #a5b4c3;'>Real-Time Air Quality Monitoring</h5>
    """, unsafe_allow_html=True)
with col_logo2:
    st.image("cpcb.png", width=150)

st.markdown("---")

# --- HIGH-RESOLUTION MAP DATA ---
@st.cache_data
def load_high_res_data():
    return pd.read_csv("data/high_res_input_sample_100.csv")

def get_pm_color(pm):
    if pm <= 60:
        return [0, 200, 0]
    elif pm <= 120:
        return [255, 165, 0]
    else:
        return [255, 0, 0]

# --- SECTION 1: High Resolution Map ---
st.markdown("### 🌏 High-Resolution PM2.5 Prediction Map")
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
    opacity=0.8,
)

view_map = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=4.5, pitch=40)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_map,
    layers=[layer_map],
    tooltip={"text": "Lat: {latitude}\nLon: {longitude}\nPM2.5: {PM2.5_Predicted}"}
))

with st.expander("📋 Show High-Resolution Prediction Table"):
    st.dataframe(df_map[["latitude", "longitude", "PM2.5_Predicted"]].round(2))

st.download_button(
    label="📅 Download High-Res Predictions",
    data=df_map.to_csv(index=False).encode(),
    file_name="pm25_high_res_predictions.csv",
    mime="text/csv"
)

# --- SECTION 2: Real-Time City Monitoring ---
st.markdown("### 🌐 Multi-City Live PM2.5 & PM10 Monitoring Dashboard")

available_cities = {
    "Delhi": "data/delhi_pm_data.csv",
    "Bangalore": "data/bangalore_pm_data.csv",
    "Hyderabad": "data/hyderabad_pm_data.csv",
    "Kolkata": "data/kolkata_pm_data.csv"
}

st.sidebar.header("🔧 Configuration")
selected_cities = st.sidebar.multiselect("Select cities to monitor:", list(available_cities.keys()), default=list(available_cities.keys()))
refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [1, 5, 10], index=1)

frames = []
for city in selected_cities:
    path = available_cities.get(city)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df["city"] = city
        frames.append(df)
    else:
        st.warning(f"Data for {city} not found at {path}")

if not frames:
    st.stop()

df_all = pd.concat(frames, ignore_index=True)
model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m", "pbl_height", "wind_speed_10m", "hour"]

# --- EMAIL ALERT FUNCTION ---
def send_alert_email(city, pm25, pm10, hour):
    recipient = email_recipients.get(city)
    if not recipient:
        return
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient
    msg["Subject"] = f"🚨 ALERT: PM Levels in {city}"
    body = f"""
    ⚠️ Air Quality Alert for {city} at Hour {hour} ⚠️

    PM2.5: {pm25:.2f} μg/m³ (Threshold: {PM25_THRESHOLD})
    PM10 : {pm10:.2f} μg/m³ (Threshold: {PM10_THRESHOLD})
    """
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email failed: {e}")

# --- LIVE LOOP ---
st.subheader("📱 Realtime AQ Monitoring")
placeholder = st.empty()
alert_sent = set()

for i in range(len(df_all)):
    row = df_all.iloc[i].copy()
    features_df = pd.DataFrame([row[model_features]])
    row["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(features_df))[0]
    row["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(features_df))[0]

    if (row["PM2.5_pred"] > PM25_THRESHOLD or row["PM10_pred"] > PM10_THRESHOLD) and row["city"] not in alert_sent:
        send_alert_email(row["city"], row["PM2.5_pred"], row["PM10_pred"], row["hour"])
        alert_sent.add(row["city"])

    with placeholder.container():
        st.markdown(f"### 🌆 {row['city']} | ⏱️ Hour: {row['hour']}")
        col1, col2 = st.columns(2)
        col1.markdown(f"<div style='padding:20px;background:#112233;color:white;border-radius:10px;'>PM2.5<br><span style='font-size:36px'>{row['PM2.5_pred']:.2f}</span></div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='padding:20px;background:#112233;color:white;border-radius:10px;'>PM10<br><span style='font-size:36px'>{row['PM10_pred']:.2f}</span></div>", unsafe_allow_html=True)

        view = pdk.ViewState(latitude=row["latitude"], longitude=row["longitude"], zoom=6, pitch=40)
        layer = pdk.Layer("ScatterplotLayer", data=pd.DataFrame([row]), get_position='[longitude, latitude]', get_fill_color=get_pm_color(row["PM2.5_pred"]), get_radius=10000)

        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/dark-v10", initial_view_state=view, layers=[layer]))

        last_10 = df_all[df_all["city"] == row["city"]].copy().tail(10)
        last_10["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(last_10[model_features]))
        last_10["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(last_10[model_features]))

        melted = pd.melt(last_10, id_vars=["hour"], value_vars=["PM2.5_pred", "PM10_pred"], var_name="Pollutant", value_name="Concentration")

        chart = alt.Chart(melted).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("hour:O", title="Hour"),
            y=alt.Y("Concentration:Q", title="μg/m³"),
            color=alt.Color("Pollutant:N"),
            tooltip=["hour", "Pollutant", "Concentration"]
        ).properties(title="📊 Last 10 Readings", height=350, width=850).interactive()

        st.altair_chart(chart, use_container_width=True)
        st.markdown("---")

    time.sleep(refresh_interval)

# --- FINAL DOWNLOAD ---
st.download_button(
    label="📆 Download All City Predictions",
    data=df_all.to_csv(index=False).encode(),
    file_name="all_city_pm_predictions.csv",
    mime="text/csv"
)

