import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

# === Load label encoders ===
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# === Load model ===
model = XGBRegressor()
model.load_model("xgb_f1_model.json")

# === Load dataset for options and defaults ===
df = pd.read_csv("f1_2018-2024_Model_data.csv")

# === UI Inputs ===
st.title("🏁 F1 Driver Position Predictor")
st.write("Input the race details below to predict the driver's finishing position.")

input_data = {}

# === Basic Race Info ===
st.subheader("📅 Basic Race Info")

# Grand Prix Name
options_grandprix = sorted(df['GrandPrix'].dropna().astype(str).unique())
selected_grandprix = st.selectbox("Select Grand Prix", options_grandprix)
encoded_grandprix = encoders['GrandPrix'].transform([selected_grandprix])[0]
input_data['GrandPrix'] = encoded_grandprix

# Country
options_country = sorted(df['Country'].dropna().astype(str).unique())
selected_country = st.selectbox("Select Country", options_country)
encoded_country = encoders['Country'].transform([selected_country])[0]
input_data['Country'] = encoded_country

# === Race Input ===
st.subheader("🏎️ Race Inputs")

# Driver
options_driver = sorted(df['Driver'].dropna().astype(str).unique())
selected_driver = st.selectbox("Select Driver", options_driver)
encoded_driver = encoders['Driver'].transform([selected_driver])[0]
input_data['Driver'] = encoded_driver

# Team
options_team = sorted(df['Team'].dropna().astype(str).unique())
selected_team = st.selectbox("Select Team", options_team)
encoded_team = encoders['Team'].transform([selected_team])[0]
input_data['Team'] = encoded_team

# Status
options_status = sorted(df['Status'].dropna().astype(str).unique())
selected_status = st.selectbox("Select Status", options_status)
encoded_status = encoders['Status'].transform([selected_status])[0]
input_data['Status'] = encoded_status

# First Compound
options_first_compound = sorted(df['FirstCompound'].dropna().astype(str).unique())
selected_first_compound = st.selectbox("Select First Compound", options_first_compound)
encoded_first_compound = encoders['FirstCompound'].transform([selected_first_compound])[0]
input_data['FirstCompound'] = encoded_first_compound

# === Race Condition ===
st.subheader("🌦️ Race Conditions")

# Grid Position
grid_position = st.slider("Grid Position", min_value=1, max_value=20, value=10, step=1)
input_data['GridPosition'] = grid_position

# Points Scored
points_scored = st.slider("Points Scored", min_value=0, max_value=50, value=10, step=1)
input_data['Points'] = points_scored

# Mean Lap Time
mean_lap_time = st.number_input("Mean Lap Time (in seconds)", value=float(df['MeanLapTime'].dropna().mean()), step=0.1)
input_data['MeanLapTime'] = mean_lap_time

# Std Lap Time
std_lap_time = st.number_input("Std Lap Time (in seconds)", value=float(df['StdLapTime'].dropna().mean()), step=0.1)
input_data['StdLapTime'] = std_lap_time

# Pit Stops
pit_stops = st.slider("Pit Stops", min_value=0, max_value=5, value=2, step=1)
input_data['PitStops'] = pit_stops

# Stint Count
stint_count = st.slider("Stint Count", min_value=1, max_value=5, value=2, step=1)
input_data['StintCount'] = stint_count

# Air Temp
air_temp = st.slider("Air Temp (°C)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
input_data['AirTemp'] = air_temp

# Rainfall
rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
input_data['Rainfall'] = rainfall

# Humidity
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
input_data['Humidity'] = humidity

# Wind Speed
wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
input_data['WindSpeed'] = wind_speed

# Track Temp
track_temp = st.slider("Track Temp (°C)", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
input_data['TrackTemp'] = track_temp

# === Predict ===
if st.button("🚦 Predict Position"):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"🏆 Predicted Finishing Position: **{round(prediction)}**")
