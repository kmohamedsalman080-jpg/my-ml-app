import streamlit as st
import joblib
import pandas as pd
import psutil

# Page Configuration
st.set_page_config(page_title="ASUS TUF AI Thermal Monitor", page_icon="🔥", layout="centered")

st.title("💻 ASUS TUF AI Thermal Monitor Dashboard")
st.markdown("---")
st.write("Welcome to your professional AI Engineering Web App! Monitor your laptop temperature using Machine Learning.")

# 1. Load the Saved AI Model
try:
    model = joblib.load("cpu_temp_model.pkl")
except FileNotFoundError:
    st.error("❌ Error: 'cpu_temp_model.pkl' not found! Please run Day 5 code first.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio("Select Operating Mode", ["Live Hardware (psutil)", "Manual Simulation Slider"])

if mode == "Live Hardware (psutil)":
    st.subheader("🔴 Live Asus TUF CPU Telemetry")
    st.info("Click the button below to fetch current system stats and predict temperature using AI.")
    
    if st.button("Fetch Live Stats"):
        cpu_load = psutil.cpu_percent(interval=1)
        input_df = pd.DataFrame([{"CPU_Load": cpu_load}])
        pred_temp = model.predict(input_df)[0]
        
        col1, col2 = st.columns(2)
        col1.metric("Live CPU Load", f"{cpu_load}%")
        col2.metric("Predicted Temp", f"{pred_temp:.2f}°C")
        
        if pred_temp < 70:
            st.success("🟢 Status: COOL & SAFE (Normal Operations)")
        elif 70 <= pred_temp <= 82:
            st.warning("🟡 Status: WARM (Heavy Tasks / Multitasking)")
        else:
            st.error("🔴 Status: HIGH TEMP ALERT! (Cooling Fan Speed Max Required)")

else:
    st.subheader("🎛️ Manual Slider Simulation")
    cpu_load = st.slider("Drag to set CPU Load (%)", 0.0, 100.0, 45.0)
    
    input_df = pd.DataFrame