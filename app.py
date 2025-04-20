import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

# === Load Resources ===
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

model = XGBRegressor()
model.load_model("xgb_f1_model.json")

df = pd.read_csv("f1_2018-2024_Model_data.csv")

# === UI ===
st.title("🏁 F1 Driver Position Predictor")
st.markdown("Enter race information to predict the driver's finishing position.")

input_data = {}

# === Basic Race Info ===
st.header("📅 Basic Race Info")

# 1. Year and Round (added back)
year = st.number_input("Year", min_value=2010, max_value=2030, value=2024, step=1)
round_ = st.number_input("Round", min_value=1, max_value=25, value=1, step=1)
input_data["Year"] = int(year)
input_data["Round"] = int(round_)

# 2. Grand Prix and Country (categorical)
for col in ['GrandPrix', 'Country']:
    options = sorted(df[col].dropna().astype(str).unique())
    selected = st.selectbox(f"Select {col}", options)
    input_data[col] = encoders[col].transform([selected])[0]

# === Race Input ===
st.header("🏎️ Race Input")

for col in ['Driver', 'Team', 'Status', 'FirstCompound']:
    options = sorted(df[col].dropna().astype(str).unique())
    selected = st.selectbox(f"Select {col}", options)
    input_data[col] = encoders[col].transform([selected])[0]

# === Race Conditions ===
st.header("🌡️ Race Conditions")

# Sliders & number inputs for numerical features
input_data["GridPosition"] = st.slider("Grid Position", 1, 20, 10)
input_data["Points"] = st.slider("Points Scored", 0, 50, 0)
input_data["MeanLapTime"] = st.number_input("Mean Lap Time", min_value=0.0, value=round(df["MeanLapTime"].mean(), 3), step=0.001)
input_data["StdLapTime"] = st.number_input("Std Lap Time", min_value=0.0, value=round(df["StdLapTime"].mean(), 3), step=0.001)
input_data["PitStops"] = st.slider("Pit Stops", 0, 5, 1)
input_data["StintCount"] = st.slider("Stint Count", 1, 5, 2)
input_data["AirTemp"] = st.slider("Air Temperature (°C)", 10.0, 50.0, float(round(df["AirTemp"].mean(), 1)))
input_data["Rainfall"] = st.slider("Rainfall", 0.0, 1.0, float(round(df["Rainfall"].mean(), 2)))
input_data["Humidity"] = st.slider("Humidity (%)", 0.0, 100.0, float(round(df["Humidity"].mean(), 1)))
input_data["WindSpeed"] = st.slider("Wind Speed (km/h)", 0.0, 20.0, float(round(df["WindSpeed"].mean(), 1)))
input_data["TrackTemp"] = st.slider("Track Temperature (°C)", 10.0, 60.0, float(round(df["TrackTemp"].mean(), 1)))

# === Prediction ===
st.markdown("---")
if st.button("🚦 Predict Position"):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.markdown(f"""
    <h2 style='text-align: center; color: #00ffcc;'>🏆 Predicted Finishing Position: <strong>{round(prediction)}</strong></h2>
    """, unsafe_allow_html=True)

