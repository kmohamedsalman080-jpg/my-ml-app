import streamlit as st
import joblib
import pandas as pd
import psutil
import time

# Page Configuration
st.set_page_config(page_title="ASUS TUF Advanced Thermal Hub", page_icon="📊", layout="wide")

st.title("📊 ASUS TUF Advanced AI Thermal Hub")
st.markdown("---")
st.write("Professional Monitoring Dashboard with Real-Time Telemetry & AI Time-Series Tracking.")

# 1. Load Model
try:
    model = joblib.load("cpu_temp_model.pkl")
except FileNotFoundError:
    st.error("❌ Error: 'cpu_temp_model.pkl' not found! Run Day 5 code first.")
    st.stop()

# 2. Sidebar Navigation
st.sidebar.header("⚙️ Dashboard Controls")
app_mode = st.sidebar.radio("Select View", ["Instant Check", "Live Stream Graph"])

if app_mode == "Instant Check":
    st.subheader("⚡ Instant Hardware Telemetry & Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        cpu_input = st.slider("Simulate CPU Load (%)", 0.0, 100.0, 50.0)
    
    input_df = pd.DataFrame([{"CPU_Load": cpu_input}])
    pred_temp = model.predict(input_df)[0]
    
    with col2:
        st.metric(label="Estimated CPU Temp", value=f"{pred_temp:.2f}°C", delta=f"{cpu_input}% Load")
        
    if pred_temp < 70:
        st.success("🟢 Status: Safe Operation")
    elif pred_temp <= 82:
        st.warning("🟡 Status: Moderate Heat - Keep an eye out")
    else:
        st.error("🔴 Status: High Heat Alert!")

else:
    st.subheader("📈 Live Continuous Monitoring Stream")
    st.info("Streaming real-time CPU data from Asus TUF hardware...")
    
    # Placeholder for dynamic updates
    chart_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    if st.button("Start Live Stream (10 Seconds)"):
        history_data = []
        
        for i in range(10):
            cpu_val = psutil.cpu_percent(interval=1)
            temp_val = model.predict(pd.DataFrame([{"CPU_Load": cpu_val}]))[0]
            
            history_data.append({"Second": f"T-{10-i}", "CPU Load (%)": cpu_val, "Predicted Temp (°C)": temp_val})
            
            # Convert to DataFrame for charting
            df_hist = pd.DataFrame(history_data)
            
            with metric_placeholder.container():
                c1, c2 = st.columns(2)
                c1.metric("Current Live Load", f"{cpu_val}%")
                c2.metric("Current AI Temp", f"{temp_val:.2f}°C")
            
            # Update line chart dynamically
            chart_placeholder.line_chart(df_hist.set_index("Second")[["CPU Load (%)", "Predicted Temp (°C)"]])
            
        st.success("✅ Live streaming session completed!")

st.markdown("---")
st.markdown("🛠️ *AI Engineering Workflow — Powered by Scikit-Learn, Psutil & Streamlit.*")