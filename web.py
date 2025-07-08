import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
import time
import os
import smtplib
import altair as alt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
st.set_page_config(page_title="Live AQ Monitoring", layout="wide")

# --- DARK MODE CSS + SIDEBAR FIXES ---
st.markdown("""
    <style>
    .main {
        background-color: #0b1725;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #08121d;
        color: white;
        border-right: 1px solid #222;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        color: white !important;
        font-weight: bold;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        font-weight: bold;
        border-radius: 8px;
    }
    .stSelectbox > label, .stMultiSelect > label {
        color: white !important;
        font-size: 16px;
    }
    h1, h2, h3, h4, .st-bb, .st-cb {
        color: #ffffff !important;
    }
    .mapboxgl-map {
        border-radius: 10px;
        border: 1px solid #444;
    }
    ::-webkit-scrollbar {
        background-color: #111;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #444;
    }
    .stButton>button {
        background-color: #1c2d44;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #334;
    }
    .stDownloadButton > button {
        background-color: #1464b4;
        color: white;
        font-weight: bold;
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
st.markdown("### 🌐 Multi-City Live PM2.5 & PM10 Monitoring Dashboard")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    pm25_model = joblib.load("models/pm25_model.pkl")
    pm10_model = joblib.load("models/pm10_model.pkl")
    pm25_scaler = joblib.load("models/pm25_scaler.pkl")
    pm10_scaler = joblib.load("models/pm10_scaler.pkl")
    return pm25_model, pm10_model, pm25_scaler, pm10_scaler

pm25_model, pm10_model, pm25_scaler, pm10_scaler = load_models()

# --- DATA CONFIG ---
available_cities = {
    "Delhi": "data/delhi_pm_data.csv",
    "Bangalore": "data/bangalore_pm_data.csv",
    "Hyderabad": "data/hyderabad_pm_data.csv",
    "Kolkata": "data/kolkata_pm_data.csv"
}

# --- SIDEBAR CONFIG ---
st.sidebar.header("🔧 Configuration")
selected_cities = st.sidebar.multiselect(
    "Select cities to monitor:", list(available_cities.keys()), default=["Delhi", "Bangalore", "Hyderabad"])
refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [1, 5, 10], index=1)

# --- LOAD DATA ---
data_frames = []
for city in selected_cities:
    path = available_cities.get(city)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df["city"] = city
        data_frames.append(df)
    else:
        st.warning(f"Data file for {city} not found at: {path}")

if not data_frames:
    st.stop()

df_all = pd.concat(data_frames, ignore_index=True)
model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m", "pbl_height", "wind_speed_10m", "hour"]

# --- EMAIL ALERT ---
def send_alert_email(city, pm25, pm10, hour):
    recipient = email_recipients.get(city)
    if not recipient:
        return
    subject = f"🚨 ALERT: Dangerous PM Levels in {city}"
    body = f"""
⚠️ Air Quality Alert for {city} at Hour {hour} ⚠️

PM2.5: {pm25:.2f} μg/m³ (Threshold: {PM25_THRESHOLD})
PM10 : {pm10:.2f} μg/m³ (Threshold: {PM10_THRESHOLD})

Please take immediate action and inform relevant authorities.
"""
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Email failed for {city}: {e}")

# --- COLOR LOGIC ---
def get_color(pm25):
    if pm25 <= 60:
        return [0, 200, 0]
    elif pm25 <= 120:
        return [255, 165, 0]
    else:
        return [255, 0, 0]

# --- LIVE MONITORING ---
st.subheader("📡 Realtime Air Quality Monitoring")
placeholder = st.empty()
alert_sent = set()

for i in range(len(df_all)):
    row = df_all.iloc[i]
    features = pd.DataFrame([row[model_features]])
    row["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(features))[0]
    row["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(features))[0]

    if (row["PM2.5_pred"] > PM25_THRESHOLD or row["PM10_pred"] > PM10_THRESHOLD) and row["city"] not in alert_sent:
        send_alert_email(row["city"], row["PM2.5_pred"], row["PM10_pred"], row["hour"])
        alert_sent.add(row["city"])

    with placeholder.container():
        st.markdown(f"### 🌆 {row['city']} | ⏱️ Hour: {row['hour']}")
        col1, col2 = st.columns(2)

        col1.markdown(f"""
            <div style='background-color:#112233; padding:20px; border-radius:10px; text-align:center; color:white; font-size:24px; font-weight:bold;'>
            PM2.5<br><span style='font-size:40px'>{row['PM2.5_pred']:.2f}</span>
            </div>""", unsafe_allow_html=True)

        col2.markdown(f"""
            <div style='background-color:#112233; padding:20px; border-radius:10px; text-align:center; color:white; font-size:24px; font-weight:bold;'>
            PM10<br><span style='font-size:40px'>{row['PM10_pred']:.2f}</span>
            </div>""", unsafe_allow_html=True)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([row]),
            get_position='[longitude, latitude]',
            get_fill_color=get_color(row["PM2.5_pred"]),
            get_radius=10000,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=row["latitude"],
            longitude=row["longitude"],
            zoom=6,
            pitch=40,
        )

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=[layer]
        ))

        # Line Chart (last 10 points)
        last_10 = df_all[df_all["city"] == row["city"]].copy().tail(10)
        last_10["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(last_10[model_features]))
        last_10["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(last_10[model_features]))

        chart_data = pd.melt(last_10, id_vars=["hour"], value_vars=["PM2.5_pred", "PM10_pred"],
                             var_name="Pollutant", value_name="Concentration")

        live_chart = alt.Chart(chart_data).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("hour:O", title="Hour", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
            y=alt.Y("Concentration:Q", title="μg/m³", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
            color=alt.Color("Pollutant:N", legend=alt.Legend(labelFontSize=14, titleFontSize=16)),
            tooltip=["hour", "Pollutant", "Concentration"]
        ).properties(
            height=350,
            width=850,
            title=alt.TitleParams(text="📊 Live Update - Last 10 Readings", fontSize=18, anchor="start")
        ).configure_axis(grid=True, gridOpacity=0.2).interactive()

        st.altair_chart(live_chart, use_container_width=True)
        st.markdown("---")

    time.sleep(refresh_interval)

# --- DOWNLOAD ---
st.download_button(
    label="📅 Download All Predictions",
    data=df_all.to_csv(index=False).encode(),
    file_name="all_city_pm_predictions.csv",
    mime="text/csv"
)

#################################################################################################################################
# import streamlit as st
# import pandas as pd
# import joblib
# import pydeck as pdk
# import time
# import os
# import smtplib
# import altair as alt
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- CONFIGURATION ---
# EMAIL_SENDER = "nikithnandi08@gmail.com"
# EMAIL_PASSWORD = "sshz jpyi pibg jxev"
# EMAIL_RECEIVER = "nikithnandi08@gmail.com"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587

# PM25_THRESHOLD = 120
# PM10_THRESHOLD = 250

# email_recipients = {
#     "Delhi": EMAIL_RECEIVER,
#     "Bangalore": EMAIL_RECEIVER,
#     "Hyderabad": EMAIL_RECEIVER,
#     "Kolkata": EMAIL_RECEIVER
# }

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Live AQ Monitoring", layout="wide")

# # --- DARK MODE CSS ---
# st.markdown("""
#     <style>
#     .main {
#         background-color: #0b1725;
#         color: #ffffff;
#     }
#     section[data-testid="stSidebar"] {
#         background-color: #08121d;
#         color: white;
#         border-right: 1px solid #222;
#     }
#     h1, h2, h3, h4, .st-bb, .st-cb {
#         color: #ffffff !important;
#     }
#     .mapboxgl-map {
#         border-radius: 10px;
#         border: 1px solid #444;
#     }
#     ::-webkit-scrollbar {
#         background-color: #111;
#     }
#     ::-webkit-scrollbar-thumb {
#         background-color: #444;
#     }
#     .stButton>button, .stSelectbox>div {
#         background-color: #1c2d44;
#         color: #ffffff;
#         border-radius: 8px;
#         border: 1px solid #334;
#     }
#     .stDownloadButton > button {
#         background-color: #1464b4;
#         color: white;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- HEADER ---
# col_logo1, col_title, col_logo2 = st.columns([1, 5, 1])
# with col_logo1:
#     st.image("ISRO-Color.png", width=150)
# with col_title:
#     st.markdown("""
#         <h2 style='text-align: center; color: #64b5f6;'>ISRO & CPCB AIR POLLUTION LIVE MONITORING SITE</h2>
#         <h5 style='text-align: center; color: #a5b4c3;'>Real-Time Air Quality Monitoring</h5>
#     """, unsafe_allow_html=True)
# with col_logo2:
#     st.image("cpcb.png", width=150)

# st.markdown("---")
# st.markdown("### 🌐 Multi-City Live PM2.5 & PM10 Monitoring Dashboard")

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     pm25_model = joblib.load("models/pm25_model.pkl")
#     pm10_model = joblib.load("models/pm10_model.pkl")
#     pm25_scaler = joblib.load("models/pm25_scaler.pkl")
#     pm10_scaler = joblib.load("models/pm10_scaler.pkl")
#     return pm25_model, pm10_model, pm25_scaler, pm10_scaler

# pm25_model, pm10_model, pm25_scaler, pm10_scaler = load_models()

# # --- CITY DATA ---
# available_cities = {
#     "Delhi": "data/delhi_pm_data.csv",
#     "Bangalore": "data/bangalore_pm_data.csv",
#     "Hyderabad": "data/hyderabad_pm_data.csv",
#     "Kolkata": "data/kolkata_pm_data.csv"
# }

# # --- SIDEBAR ---
# st.sidebar.header("🔧 Configuration")
# selected_cities = st.sidebar.multiselect(
#     "Select cities to monitor:", list(available_cities.keys()), default=["Delhi", "Bangalore", "Hyderabad"])
# refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [1, 5, 10], index=0)

# # --- LOAD DATA ---
# data_frames = []
# for city in selected_cities:
#     path = available_cities.get(city)
#     if path and os.path.exists(path):
#         df = pd.read_csv(path)
#         df["city"] = city
#         data_frames.append(df)
#     else:
#         st.warning(f"Data file for {city} not found at: {path}")

# if not data_frames:
#     st.stop()

# df_all = pd.concat(data_frames, ignore_index=True)

# model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
#                   "pbl_height", "wind_speed_10m", "hour"]

# # --- EMAIL ---
# def send_alert_email(city, pm25, pm10, hour):
#     recipient = email_recipients.get(city)
#     if not recipient:
#         return
#     subject = f"🚨 ALERT: Dangerous PM Levels in {city}"
#     body = f"""
# ️ Air Quality Alert for {city} at Hour {hour} ️

# PM2.5: {pm25:.2f} μg/m³ (Threshold: {PM25_THRESHOLD})
# PM10 : {pm10:.2f} μg/m³ (Threshold: {PM10_THRESHOLD})

# Please take immediate action and inform relevant authorities.

# -- Automated Monitoring System
# """
#     msg = MIMEMultipart()
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = recipient
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))
#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.send_message(msg)
#             print(f"✅ Email alert sent for {city}")
#     except Exception as e:
#         print(f"❌ Error sending email for {city}: {e}")

# # --- COLOR FUNCTION ---
# def get_color(pm25):
#     if pm25 <= 60:
#         return [0, 200, 0]
#     elif pm25 <= 120:
#         return [255, 165, 0]
#     else:
#         return [255, 0, 0]

# # --- MONITORING ---
# st.subheader("🛱️ Realtime Air Quality Monitoring")
# placeholder = st.empty()
# alert_sent = set()

# for i in range(len(df_all)):
#     row = df_all.iloc[i]
#     features = pd.DataFrame([row[model_features]])
#     scaled_pm25 = pm25_scaler.transform(features)
#     scaled_pm10 = pm10_scaler.transform(features)
#     row["PM2.5_pred"] = pm25_model.predict(scaled_pm25)[0]
#     row["PM10_pred"] = pm10_model.predict(scaled_pm10)[0]

#     if (row["PM2.5_pred"] > PM25_THRESHOLD or row["PM10_pred"] > PM10_THRESHOLD) and row["city"] not in alert_sent:
#         send_alert_email(row["city"], row["PM2.5_pred"], row["PM10_pred"], row["hour"])
#         alert_sent.add(row["city"])

#     with placeholder.container():
#         st.markdown(f"### 🏖️ {row['city']} | ⏱️ Hour: {row['hour']}")
#         col1, col2 = st.columns(2)

#         col1.markdown(f"""
#             <div style='background-color:#112233; padding:20px; border-radius:10px; text-align:center; color:white; font-size:24px; font-weight:bold;'>
#             PM2.5<br><span style='font-size:40px'>{row['PM2.5_pred']:.2f}</span>
#             </div>""", unsafe_allow_html=True)

#         col2.markdown(f"""
#             <div style='background-color:#112233; padding:20px; border-radius:10px; text-align:center; color:white; font-size:24px; font-weight:bold;'>
#             PM10<br><span style='font-size:40px'>{row['PM10_pred']:.2f}</span>
#             </div>""", unsafe_allow_html=True)

#         layer = pdk.Layer(
#             "ScatterplotLayer",
#             data=pd.DataFrame([row]),
#             get_position='[longitude, latitude]',
#             get_fill_color=get_color(row["PM2.5_pred"]),
#             get_radius=10000,
#             pickable=True,
#         )

#         view_state = pdk.ViewState(
#             latitude=row["latitude"],
#             longitude=row["longitude"],
#             zoom=6,
#             pitch=40,
#         )

#         st.pydeck_chart(pdk.Deck(
#             map_style="mapbox://styles/mapbox/dark-v10",
#             initial_view_state=view_state,
#             layers=[layer]
#         ))

#         last_10 = df_all[df_all["city"] == row["city"]].copy().tail(10)
#         if not last_10.empty:
#             last_10["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(last_10[model_features]))
#             last_10["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(last_10[model_features]))

#             live_melted = pd.melt(
#                 last_10,
#                 id_vars=["hour"],
#                 value_vars=["PM2.5_pred", "PM10_pred"],
#                 var_name="Pollutant",
#                 value_name="Concentration"
#             )

#             live_chart = alt.Chart(live_melted).mark_line(point=True, strokeWidth=3).encode(
#                 x=alt.X("hour:O", title="Hour", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
#                 y=alt.Y("Concentration:Q", title="μg/m³", axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
#                 color=alt.Color("Pollutant:N", legend=alt.Legend(labelFontSize=14, titleFontSize=16)),
#                 tooltip=["hour", "Pollutant", "Concentration"]
#             ).properties(
#                 height=350,
#                 width=850,
#                 title=alt.TitleParams(text="📊 Live Update - Last 10 Readings", fontSize=18, anchor="start")
#             ).configure_axis(grid=True, gridOpacity=0.2).interactive()

#             st.altair_chart(live_chart, use_container_width=True)

#         st.markdown("---")
#     time.sleep(refresh_interval)

# # --- DOWNLOAD ---
# st.download_button(
#     label="🗕️ Download All Predictions",
#     data=df_all.to_csv(index=False).encode(),
#     file_name="all_city_pm_predictions.csv",
#     mime="text/csv"
# )

###################################################################################################################################
# import streamlit as st
# import pandas as pd
# import pydeck as pdk
# import time
# from datetime import datetime

# # Set page config
# st.set_page_config(page_title="Live PM Monitoring Dashboard", layout="wide")

# # Sidebar UI
# st.sidebar.header("📊 Choose Module")
# module = st.sidebar.selectbox("Select Module", ["Live PM Monitoring"])
# refresh = st.sidebar.slider("⏱ Refresh Interval (sec)", 1, 30, 5)
# data_source = st.sidebar.radio("📡 Data Source", ["Simulated", "Real"], index=0)

# # Function to classify PM2.5 AQI levels
# def get_pm25_quality(value):
#     if value <= 50: return "🟢 Good"
#     elif value <= 100: return "🟡 Moderate"
#     elif value <= 150: return "🟠 Unhealthy (Sensitive)"
#     elif value <= 200: return "🔴 Unhealthy"
#     else: return "⚫ Hazardous"

# # Load simulated data
# def load_simulated_data():
#     df = pd.read_csv("data/sample_pm_predictions_200.csv")
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     latest_time = df['timestamp'].max()
#     return df[df['timestamp'] == latest_time]

# # Real-time placeholder (optional)
# def load_real_data():
#     # Placeholder for future real-time integration (API, sensors)
#     st.warning("⚠️ Real-time data source not yet connected. Showing simulated data.")
#     return load_simulated_data()

# # Load data based on source
# if data_source == "Simulated":
#     df = load_simulated_data()
# else:
#     df = load_real_data()

# # Main Dashboard Title
# st.title("🌫 Live Monitoring of PM2.5 & PM10")
# latest_time = df['timestamp'].iloc[0]
# st.subheader(f"🕒 Timestamp: {latest_time}")

# # Metric Display
# col1, col2, col3 = st.columns(3)
# with col1:
#     pm25 = df['pm25_pred'].mean()
#     st.metric("PM2.5 (µg/m³)", f"{pm25:.2f}", get_pm25_quality(pm25))
# with col2:
#     pm10 = df['pm10_pred'].mean()
#     st.metric("PM10 (µg/m³)", f"{pm10:.2f}")
# with col3:
#     st.metric("No. of Stations", len(df))

# # Pydeck Map
# st.markdown("### 🗺 PM Concentration Map")
# st.pydeck_chart(pdk.Deck(
#     initial_view_state=pdk.ViewState(
#         latitude=df['latitude'].mean(),
#         longitude=df['longitude'].mean(),
#         zoom=4,
#         pitch=30
#     ),
#     layers=[
#         pdk.Layer(
#             "ScatterplotLayer",
#             data=df,
#             get_position='[longitude, latitude]',
#             get_fill_color='[pm25_pred * 2, 255 - pm25_pred, 100, 160]',
#             get_radius=20000,
#             pickable=True
#         )
#     ],
#     tooltip={"text": "Lat: {latitude}, Lon: {longitude}\nPM2.5: {pm25_pred:.1f} µg/m³\nPM10: {pm10_pred:.1f} µg/m³"}
# ))

# # Raw Data Viewer
# st.markdown("### 📄 Raw Data")
# st.dataframe(df, use_container_width=True)

# # Auto Refresh
# time.sleep(refresh)
# st.experimental_rerun()
###################################################################################################################################

# import streamlit as st
# import pandas as pd
# import joblib
# import pydeck as pdk
# import time

# # Load models and scalers
# @st.cache_resource
# def load_models():
#     pm25_model = joblib.load("models/pm25_model.pkl")
#     pm10_model = joblib.load("models/pm10_model.pkl")
#     pm25_scaler = joblib.load("models/pm25_scaler.pkl")
#     pm10_scaler = joblib.load("models/pm10_scaler.pkl")
#     return pm25_model, pm10_model, pm25_scaler, pm10_scaler

# pm25_model, pm10_model, pm25_scaler, pm10_scaler = load_models()

# st.title("🌍 Real-Time Air Quality Monitoring (PM2.5 & PM10)")

# # Load dataset
# @st.cache_data
# def load_data():
#     df = pd.read_csv("data/sample_pm_prediction_data_final.csv")
#     return df

# df = load_data()

# # Model features
# model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m", "pbl_height", "wind_speed_10m", "hour"]
# required_columns = ["city", "latitude", "longitude"] + model_features

# if not all(col in df.columns for col in required_columns):
#     st.error("The dataset is missing required columns.")
#     st.stop()

# # Setup UI
# st.subheader("📡 Simulating Real-Time Predictions")
# placeholder = st.empty()

# for idx, row in df.iterrows():
#     # Prepare input features
#     features = row[model_features].values.reshape(1, -1)

#     # Scale and predict
#     pm25_input = pm25_scaler.transform(features)
#     pm10_input = pm10_scaler.transform(features)
#     pm25_pred = pm25_model.predict(pm25_input)[0]
#     pm10_pred = pm10_model.predict(pm10_input)[0]

#     def get_color(pm25):
#         if pm25 <= 60:
#             return [0, 255, 0]  # Green
#         elif pm25 <= 120:
#             return [255, 255, 0]  # Yellow
#         else:
#             return [255, 0, 0]  # Red

#     # Map Layer
#     map_data = pd.DataFrame([{
#         'city': row['city'],
#         'latitude': row['latitude'],
#         'longitude': row['longitude'],
#         'pm25': pm25_pred
#     }])

#     layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=map_data,
#         get_position='[longitude, latitude]',
#         get_fill_color=get_color(pm25_pred),
#         get_radius=10000,
#         pickable=True
#     )

#     view_state = pdk.ViewState(
#         latitude=row['latitude'],
#         longitude=row['longitude'],
#         zoom=6,
#         pitch=30
#     )

#     # Display dashboard
#     with placeholder.container():
#         st.write(f"### 📍 Monitoring: {row['city']}")
#         st.metric("PM2.5", f"{pm25_pred:.2f} µg/m³")
#         st.metric("PM10", f"{pm10_pred:.2f} µg/m³")
#         st.pydeck_chart(pdk.Deck(map_style='mapbox://styles/mapbox/light-v9',
#                                  initial_view_state=view_state,
#                                  layers=[layer]))
#         st.write("---")
#         time.sleep(1)  # Delay to simulate live monitoring

# # Done
# st.success("✅ Real-time simulation complete.")
##################################################################################################################################

# import streamlit as st
# import pandas as pd
# import joblib
# import pydeck as pdk
# import time
# import os
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- CONFIGURATION ---
# EMAIL_SENDER = "nikithnandi08@gmail.com"
# EMAIL_PASSWORD = "sshz jpyi pibg jxev"  # Gmail App Password
# EMAIL_RECEIVER = "nikithnandi08@gmail.com"  # Recipient email

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587

# PM25_THRESHOLD = 120
# PM10_THRESHOLD = 250

# email_recipients = {
#     "Delhi": EMAIL_RECEIVER,
#     "Bangalore": EMAIL_RECEIVER,
#     "Hyderabad": EMAIL_RECEIVER,
#     "Kolkata": EMAIL_RECEIVER
# }

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Live AQ Monitoring", layout="wide")
# st.title("🌐 Multi-City Live PM2.5 & PM10 Monitoring Dashboard")

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     pm25_model = joblib.load("models/pm25_model.pkl")
#     pm10_model = joblib.load("models/pm10_model.pkl")
#     pm25_scaler = joblib.load("models/pm25_scaler.pkl")
#     pm10_scaler = joblib.load("models/pm10_scaler.pkl")
#     return pm25_model, pm10_model, pm25_scaler, pm10_scaler

# pm25_model, pm10_model, pm25_scaler, pm10_scaler = load_models()

# # --- CITY DATA CONFIG ---
# available_cities = {
#     "Delhi": "data/delhi_pm_data.csv",
#     "Bangalore": "data/bangalore_pm_data.csv",
#     "Hyderabad": "data/hyderabad_pm_data.csv",
#     "Kolkata": "data/kolkata_pm_data.csv"
# }

# # --- SIDEBAR CONFIG ---
# st.sidebar.header("🔧 Configuration")
# selected_cities = st.sidebar.multiselect(
#     "Select cities to monitor:", list(available_cities.keys()), default=["Delhi", "Bangalore", "Hyderabad"])
# refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [1, 5, 10], index=0)

# # --- LOAD CITY DATA ---
# data_frames = []
# for city in selected_cities:
#     path = available_cities.get(city)
#     if path and os.path.exists(path):
#         df = pd.read_csv(path)
#         df["city"] = city
#         data_frames.append(df)
#     else:
#         st.warning(f"Data file for {city} not found at: {path}")

# if not data_frames:
#     st.stop()

# df_all = pd.concat(data_frames, ignore_index=True)

# # --- FEATURES ---
# model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
#                   "pbl_height", "wind_speed_10m", "hour"]

# # --- EMAIL ALERT FUNCTION ---
# def send_alert_email(city, pm25, pm10, hour):
#     recipient = email_recipients.get(city)
#     if not recipient:
#         return

#     subject = f"🚨 ALERT: Dangerous PM Levels in {city}"
#     body = f"""
# ⚠️ Air Quality Alert for {city} at Hour {hour} ⚠️

# PM2.5: {pm25:.2f} μg/m³ (Threshold: {PM25_THRESHOLD})
# PM10 : {pm10:.2f} μg/m³ (Threshold: {PM10_THRESHOLD})

# Please take immediate action and inform relevant authorities.

# -- Automated Monitoring System
# """

#     msg = MIMEMultipart()
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = recipient
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))

#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.send_message(msg)
#             print(f"✅ Email alert sent for {city}")
#     except Exception as e:
#         print(f"❌ Error sending email for {city}: {e}")

# # --- COLOR FUNCTION ---
# def get_color(pm25):
#     if pm25 <= 60:
#         return [0, 200, 0]
#     elif pm25 <= 120:
#         return [255, 165, 0]
#     else:
#         return [255, 0, 0]

# # --- DISPLAY ---
# st.subheader("📡 Realtime Air Quality Monitoring")
# placeholder = st.empty()
# alert_sent = set()

# # --- MONITORING LOOP ---
# for i in range(len(df_all)):
#     row = df_all.iloc[i]
#     features = pd.DataFrame([row[model_features]])

#     scaled_pm25 = pm25_scaler.transform(features)
#     scaled_pm10 = pm10_scaler.transform(features)

#     row["PM2.5_pred"] = pm25_model.predict(scaled_pm25)[0]
#     row["PM10_pred"] = pm10_model.predict(scaled_pm10)[0]

#     # --- ALERT CONDITION ---
#     if (
#         row["PM2.5_pred"] > PM25_THRESHOLD or row["PM10_pred"] > PM10_THRESHOLD
#     ) and row["city"] not in alert_sent:
#         send_alert_email(row["city"], row["PM2.5_pred"], row["PM10_pred"], row["hour"])
#         alert_sent.add(row["city"])

#     # --- UI ---
#     with placeholder.container():
#         st.markdown(f"### 🌆 {row['city']} | ⏱️ Hour: {row['hour']}")
#         col1, col2 = st.columns(2)
#         col1.metric("PM2.5", f"{row['PM2.5_pred']:.2f}")
#         col2.metric("PM10", f"{row['PM10_pred']:.2f}")

#         layer = pdk.Layer(
#             "ScatterplotLayer",
#             data=pd.DataFrame([row]),
#             get_position='[longitude, latitude]',
#             get_fill_color=get_color(row["PM2.5_pred"]),
#             get_radius=10000,
#             pickable=True,
#         )

#         view_state = pdk.ViewState(
#             latitude=row["latitude"],
#             longitude=row["longitude"],
#             zoom=6,
#             pitch=40,
#         )

#         st.pydeck_chart(pdk.Deck(
#             map_style="mapbox://styles/mapbox/light-v9",
#             initial_view_state=view_state,
#             layers=[layer]
#         ))
#         st.markdown("---")

#     time.sleep(refresh_interval)

# # --- DOWNLOAD BUTTON ---
# st.download_button(
#     label="📅 Download All Predictions",
#     data=df_all.to_csv(index=False).encode(),
#     file_name="all_city_pm_predictions.csv",
#     mime="text/csv"
# )
#####################################################################################################################################3
# import streamlit as st
# import pandas as pd
# import joblib
# import pydeck as pdk
# import time
# import os
# import smtplib
# import altair as alt
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- CONFIGURATION ---
# EMAIL_SENDER = "nikithnandi08@gmail.com"
# EMAIL_PASSWORD = "sshz jpyi pibg jxev"
# EMAIL_RECEIVER = "nikithnandi08@gmail.com"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587

# PM25_THRESHOLD = 120
# PM10_THRESHOLD = 250

# email_recipients = {
#     "Delhi": EMAIL_RECEIVER,
#     "Bangalore": EMAIL_RECEIVER,
#     "Hyderabad": EMAIL_RECEIVER,
#     "Kolkata": EMAIL_RECEIVER
# }

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Live AQ Monitoring", layout="wide")
# st.title("🌐 Multi-City Live PM2.5 & PM10 Monitoring Dashboard")

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     pm25_model = joblib.load("models/pm25_model.pkl")
#     pm10_model = joblib.load("models/pm10_model.pkl")
#     pm25_scaler = joblib.load("models/pm25_scaler.pkl")
#     pm10_scaler = joblib.load("models/pm10_scaler.pkl")
#     return pm25_model, pm10_model, pm25_scaler, pm10_scaler

# pm25_model, pm10_model, pm25_scaler, pm10_scaler = load_models()

# # --- CITY DATA CONFIG ---
# available_cities = {
#     "Delhi": "data/delhi_pm_data.csv",
#     "Bangalore": "data/bangalore_pm_data.csv",
#     "Hyderabad": "data/hyderabad_pm_data.csv",
#     "Kolkata": "data/kolkata_pm_data.csv"
# }

# # --- SIDEBAR CONFIG ---
# st.sidebar.header("🔧 Configuration")
# selected_cities = st.sidebar.multiselect(
#     "Select cities to monitor:", list(available_cities.keys()), default=["Delhi", "Bangalore", "Hyderabad"])
# refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [1, 5, 10], index=0)

# # --- LOAD CITY DATA ---
# data_frames = []
# for city in selected_cities:
#     path = available_cities.get(city)
#     if path and os.path.exists(path):
#         df = pd.read_csv(path)
#         df["city"] = city
#         data_frames.append(df)
#     else:
#         st.warning(f"Data file for {city} not found at: {path}")

# if not data_frames:
#     st.stop()

# df_all = pd.concat(data_frames, ignore_index=True)

# # --- FEATURES ---
# model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
#                   "pbl_height", "wind_speed_10m", "hour"]

# # --- EMAIL ALERT FUNCTION ---
# def send_alert_email(city, pm25, pm10, hour):
#     recipient = email_recipients.get(city)
#     if not recipient:
#         return

#     subject = f"🚨 ALERT: Dangerous PM Levels in {city}"
#     body = f"""
# ⚠️ Air Quality Alert for {city} at Hour {hour} ⚠️

# PM2.5: {pm25:.2f} μg/m³ (Threshold: {PM25_THRESHOLD})
# PM10 : {pm10:.2f} μg/m³ (Threshold: {PM10_THRESHOLD})

# Please take immediate action and inform relevant authorities.

# -- Automated Monitoring System
# """

#     msg = MIMEMultipart()
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = recipient
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))

#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.send_message(msg)
#             print(f"✅ Email alert sent for {city}")
#     except Exception as e:
#         print(f"❌ Error sending email for {city}: {e}")

# # --- COLOR FUNCTION ---
# def get_color(pm25):
#     if pm25 <= 60:
#         return [0, 200, 0]
#     elif pm25 <= 120:
#         return [255, 165, 0]
#     else:
#         return [255, 0, 0]

# # --- DISPLAY ---
# st.subheader("📡 Realtime Air Quality Monitoring")
# placeholder = st.empty()
# alert_sent = set()

# # --- MONITORING LOOP ---
# for i in range(len(df_all)):
#     row = df_all.iloc[i]
#     features = pd.DataFrame([row[model_features]])

#     scaled_pm25 = pm25_scaler.transform(features)
#     scaled_pm10 = pm10_scaler.transform(features)

#     row["PM2.5_pred"] = pm25_model.predict(scaled_pm25)[0]
#     row["PM10_pred"] = pm10_model.predict(scaled_pm10)[0]

#     # --- ALERT CONDITION ---
#     if (
#         row["PM2.5_pred"] > PM25_THRESHOLD or row["PM10_pred"] > PM10_THRESHOLD
#     ) and row["city"] not in alert_sent:
#         send_alert_email(row["city"], row["PM2.5_pred"], row["PM10_pred"], row["hour"])
#         alert_sent.add(row["city"])

#     # --- UI ---
#     with placeholder.container():
#         st.markdown(f"### 🌆 {row['city']} | ⏱️ Hour: {row['hour']}")
#         col1, col2 = st.columns(2)
#         col1.metric("PM2.5", f"{row['PM2.5_pred']:.2f}")
#         col2.metric("PM10", f"{row['PM10_pred']:.2f}")

#         layer = pdk.Layer(
#             "ScatterplotLayer",
#             data=pd.DataFrame([row]),
#             get_position='[longitude, latitude]',
#             get_fill_color=get_color(row["PM2.5_pred"]),
#             get_radius=10000,
#             pickable=True,
#         )

#         view_state = pdk.ViewState(
#             latitude=row["latitude"],
#             longitude=row["longitude"],
#             zoom=6,
#             pitch=40,
#         )

#         st.pydeck_chart(pdk.Deck(
#             map_style="mapbox://styles/mapbox/light-v9",
#             initial_view_state=view_state,
#             layers=[layer]
#         ))

#         # --- LIVE TREND PREVIEW (last 10 records) ---
#         last_10 = df_all[df_all["city"] == row["city"]].copy().tail(10)
#         if not last_10.empty:
#             last_10["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(last_10[model_features]))
#             last_10["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(last_10[model_features]))

#             live_melted = pd.melt(
#                 last_10,
#                 id_vars=["hour"],
#                 value_vars=["PM2.5_pred", "PM10_pred"],
#                 var_name="Pollutant",
#                 value_name="Concentration"
#             )

#             live_chart = alt.Chart(live_melted).mark_line(point=True).encode(
#                 x="hour:O",
#                 y="Concentration:Q",
#                 color="Pollutant:N",
#                 tooltip=["hour", "Pollutant", "Concentration"]
#             ).properties(
#                 height=200,
#                 title="📊 Live Update - Last 10 Readings"
#             ).interactive()

#             st.altair_chart(live_chart, use_container_width=True)

#         st.markdown("---")

#     time.sleep(refresh_interval)

# # --- FULL HISTORICAL TREND SECTION ---
# st.markdown("## 📈 PM2.5 and PM10 Trends")

# for city in selected_cities:
#     city_df = df_all[df_all["city"] == city].copy()
#     if city_df.empty:
#         continue

#     features = city_df[model_features]
#     scaled_pm25 = pm25_scaler.transform(features)
#     scaled_pm10 = pm10_scaler.transform(features)

#     city_df["PM2.5_pred"] = pm25_model.predict(scaled_pm25)
#     city_df["PM10_pred"] = pm10_model.predict(scaled_pm10)

#     city_melted = pd.melt(
#         city_df,
#         id_vars=["hour"],
#         value_vars=["PM2.5_pred", "PM10_pred"],
#         var_name="Pollutant",
#         value_name="Concentration"
#     )

#     with st.expander(f"📍 Trends for {city}"):
#         line_chart = alt.Chart(city_melted).mark_line(point=True).encode(
#             x=alt.X("hour:O", title="Hour"),
#             y=alt.Y("Concentration:Q", title="Concentration (μg/m³)"),
#             color="Pollutant:N",
#             tooltip=["hour", "Pollutant", "Concentration"]
#         ).properties(
#             width=700,
#             height=300,
#             title=f"{city} - PM2.5 & PM10 Trends"
#         ).interactive()

#         st.altair_chart(line_chart, use_container_width=True)

# # --- DOWNLOAD BUTTON ---
# st.download_button(
#     label="📅 Download All Predictions",
#     data=df_all.to_csv(index=False).encode(),
#     file_name="all_city_pm_predictions.csv",
#     mime="text/csv"
# )
#################################################################################################################################
# import streamlit as st
# import pandas as pd
# import joblib
# import pydeck as pdk
# import time
# import os
# import smtplib
# import altair as alt
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- CONFIGURATION ---
# EMAIL_SENDER = "nikithnandi08@gmail.com"
# EMAIL_PASSWORD = "sshz jpyi pibg jxev"
# EMAIL_RECEIVER = "nikithnandi08@gmail.com"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587

# PM25_THRESHOLD = 120
# PM10_THRESHOLD = 250

# email_recipients = {
#     "Delhi": EMAIL_RECEIVER,
#     "Bangalore": EMAIL_RECEIVER,
#     "Hyderabad": EMAIL_RECEIVER,
#     "Kolkata": EMAIL_RECEIVER
# }

# # --- PAGE CONFIG ---
# st.set_page_config(page_title="Live AQ Monitoring", layout="wide")

# # --- DARK MODE CSS ---
# st.markdown("""
#     <style>
#     .main {
#         background-color: #0b1725;
#         color: #ffffff;
#     }
#     section[data-testid="stSidebar"] {
#         background-color: #08121d;
#         color: white;
#         border-right: 1px solid #222;
#     }
#     .stMetric {
#         font-weight: bold;
#         color: white !important;
#         background-color: #112233;
#         padding: 10px;
#         border-radius: 10px;
#     }
#     h1, h2, h3, h4 {
#         color: #ffffff;
#     }
#     .st-bb, .st-cb {
#         color: #cfd8dc !important;
#     }
#     .vega-embed {
#         background-color: #0b1725 !important;
#     }
#     .mapboxgl-map {
#         border-radius: 10px;
#         border: 1px solid #444;
#     }
#     ::-webkit-scrollbar {
#         background-color: #111;
#     }
#     ::-webkit-scrollbar-thumb {
#         background-color: #444;
#     }
#     .stButton>button, .stSelectbox>div {
#         background-color: #1c2d44;
#         color: #ffffff;
#         border-radius: 8px;
#         border: 1px solid #334;
#     }
#     .stDownloadButton > button {
#         background-color: #1464b4;
#         color: white;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- HEADER UI ---
# col_logo1, col_title, col_logo2 = st.columns([1, 5, 1])
# with col_logo1:
#     st.image("ISRO-Color.png", width=150)
# with col_title:
#     st.markdown("""
#         <h2 style='text-align: center; color: #64b5f6;'>ISRO & CPCB AIR POLLUTION LIVE MONITORING SITE</h2>
#         <h5 style='text-align: center; color: #a5b4c3;'>Real-Time Air Quality Monitoring</h5>
#     """, unsafe_allow_html=True)
# with col_logo2:
#     st.image("cpcb.png", width=150)

# st.markdown("---")
# st.markdown("### 🌐 Multi-City Live PM2.5 & PM10 Monitoring Dashboard")

# # --- LOAD MODELS ---
# @st.cache_resource
# def load_models():
#     pm25_model = joblib.load("models/pm25_model.pkl")
#     pm10_model = joblib.load("models/pm10_model.pkl")
#     pm25_scaler = joblib.load("models/pm25_scaler.pkl")
#     pm10_scaler = joblib.load("models/pm10_scaler.pkl")
#     return pm25_model, pm10_model, pm25_scaler, pm10_scaler

# pm25_model, pm10_model, pm25_scaler, pm10_scaler = load_models()

# # --- CITY DATA CONFIG ---
# available_cities = {
#     "Delhi": "data/delhi_pm_data.csv",
#     "Bangalore": "data/bangalore_pm_data.csv",
#     "Hyderabad": "data/hyderabad_pm_data.csv",
#     "Kolkata": "data/kolkata_pm_data.csv"
# }

# # --- SIDEBAR CONFIG ---
# st.sidebar.header("🔧 Configuration")
# selected_cities = st.sidebar.multiselect(
#     "Select cities to monitor:", list(available_cities.keys()), default=["Delhi", "Bangalore", "Hyderabad"])
# refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [1, 5, 10], index=0)

# # --- LOAD CITY DATA ---
# data_frames = []
# for city in selected_cities:
#     path = available_cities.get(city)
#     if path and os.path.exists(path):
#         df = pd.read_csv(path)
#         df["city"] = city
#         data_frames.append(df)
#     else:
#         st.warning(f"Data file for {city} not found at: {path}")

# if not data_frames:
#     st.stop()

# df_all = pd.concat(data_frames, ignore_index=True)

# model_features = ["aod", "reflectance_SWIR", "temperature_2m", "humidity_2m",
#                   "pbl_height", "wind_speed_10m", "hour"]

# # --- EMAIL FUNCTION ---
# def send_alert_email(city, pm25, pm10, hour):
#     recipient = email_recipients.get(city)
#     if not recipient:
#         return
#     subject = f"🚨 ALERT: Dangerous PM Levels in {city}"
#     body = f"""
# ⚠️ Air Quality Alert for {city} at Hour {hour} ⚠️

# PM2.5: {pm25:.2f} μg/m³ (Threshold: {PM25_THRESHOLD})
# PM10 : {pm10:.2f} μg/m³ (Threshold: {PM10_THRESHOLD})

# Please take immediate action and inform relevant authorities.

# -- Automated Monitoring System
# """
#     msg = MIMEMultipart()
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = recipient
#     msg["Subject"] = subject
#     msg.attach(MIMEText(body, "plain"))
#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.send_message(msg)
#             print(f"✅ Email alert sent for {city}")
#     except Exception as e:
#         print(f"❌ Error sending email for {city}: {e}")

# # --- COLOR FUNCTION ---
# def get_color(pm25):
#     if pm25 <= 60:
#         return [0, 200, 0]
#     elif pm25 <= 120:
#         return [255, 165, 0]
#     else:
#         return [255, 0, 0]

# # --- MONITORING ---
# st.subheader("📡 Realtime Air Quality Monitoring")
# placeholder = st.empty()
# alert_sent = set()

# for i in range(len(df_all)):
#     row = df_all.iloc[i]
#     features = pd.DataFrame([row[model_features]])
#     scaled_pm25 = pm25_scaler.transform(features)
#     scaled_pm10 = pm10_scaler.transform(features)
#     row["PM2.5_pred"] = pm25_model.predict(scaled_pm25)[0]
#     row["PM10_pred"] = pm10_model.predict(scaled_pm10)[0]

#     if (row["PM2.5_pred"] > PM25_THRESHOLD or row["PM10_pred"] > PM10_THRESHOLD) and row["city"] not in alert_sent:
#         send_alert_email(row["city"], row["PM2.5_pred"], row["PM10_pred"], row["hour"])
#         alert_sent.add(row["city"])

#     with placeholder.container():
#         st.markdown(f"### 🌆 {row['city']} | ⏱️ Hour: {row['hour']}")
#         col1, col2 = st.columns(2)
#         col1.metric("PM2.5", f"{row['PM2.5_pred']:.2f}")
#         col2.metric("PM10", f"{row['PM10_pred']:.2f}")

#         layer = pdk.Layer(
#             "ScatterplotLayer",
#             data=pd.DataFrame([row]),
#             get_position='[longitude, latitude]',
#             get_fill_color=get_color(row["PM2.5_pred"]),
#             get_radius=10000,
#             pickable=True,
#         )

#         view_state = pdk.ViewState(
#             latitude=row["latitude"],
#             longitude=row["longitude"],
#             zoom=6,
#             pitch=40,
#         )

#         st.pydeck_chart(pdk.Deck(
#             map_style="mapbox://styles/mapbox/dark-v10",
#             initial_view_state=view_state,
#             layers=[layer]
#         ))

#         last_10 = df_all[df_all["city"] == row["city"]].copy().tail(10)
#         if not last_10.empty:
#             last_10["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(last_10[model_features]))
#             last_10["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(last_10[model_features]))

#             live_melted = pd.melt(
#                 last_10,
#                 id_vars=["hour"],
#                 value_vars=["PM2.5_pred", "PM10_pred"],
#                 var_name="Pollutant",
#                 value_name="Concentration"
#             )

#             live_chart = alt.Chart(live_melted).mark_line(point=True).encode(
#                 x="hour:O",
#                 y="Concentration:Q",
#                 color="Pollutant:N",
#                 tooltip=["hour", "Pollutant", "Concentration"]
#             ).properties(
#                 height=200,
#                 title="📊 Live Update - Last 10 Readings"
#             ).interactive()

#             st.altair_chart(live_chart, use_container_width=True)

#         st.markdown("---")

#     time.sleep(refresh_interval)

# # --- FULL TREND SECTION ---
# st.markdown("## 📈 PM2.5 and PM10 Trends")
# for city in selected_cities:
#     city_df = df_all[df_all["city"] == city].copy()
#     if city_df.empty:
#         continue

#     features = city_df[model_features]
#     city_df["PM2.5_pred"] = pm25_model.predict(pm25_scaler.transform(features))
#     city_df["PM10_pred"] = pm10_model.predict(pm10_scaler.transform(features))

#     city_melted = pd.melt(
#         city_df,
#         id_vars=["hour"],
#         value_vars=["PM2.5_pred", "PM10_pred"],
#         var_name="Pollutant",
#         value_name="Concentration"
#     )

#     with st.expander(f"📍 Trends for {city}"):
#         line_chart = alt.Chart(city_melted).mark_line(point=True).encode(
#             x=alt.X("hour:O", title="Hour"),
#             y=alt.Y("Concentration:Q", title="Concentration (μg/m³)"),
#             color="Pollutant:N",
#             tooltip=["hour", "Pollutant", "Concentration"]
#         ).properties(
#             width=700,
#             height=300,
#             title=f"{city} - PM2.5 & PM10 Trends"
#         ).interactive()

#         st.altair_chart(line_chart, use_container_width=True)

# # --- DOWNLOAD BUTTON ---
# st.download_button(
#     label="📅 Download All Predictions",
#     data=df_all.to_csv(index=False).encode(),
#     file_name="all_city_pm_predictions.csv",
#     mime="text/csv"
# )
##################################################################################################################################
