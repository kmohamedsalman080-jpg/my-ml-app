import time
import joblib
import pandas as pd
import psutil  # லேப்டாப் CPU விவரங்களை எடுக்க உதவும் லைப்ரரி

def live_thermal_monitor():
    try:
        # 1. முந்தைய நாள்களில் சேமித்த AI Model-ஐ லோட் செய்கிறோம்
        model = joblib.load("cpu_temp_model.pkl")
        print("==================================================")
        print("   🔥 ASUS TUF LIVE CPU THERMAL MONITOR (AI)     ")
        print("==================================================")
        print("Monitoring real-time CPU... Press Ctrl+C to stop.\n")
        
        while True:
            # 2. லேப்டாப்பின் உண்மையான CPU Load-ஐ பெறுகிறோம் (1 வினாடி இடைவெளியில்)
            current_cpu_load = psutil.cpu_percent(interval=1)
            
            # 3. நமது AI மாடலுக்கு இன்புட்டாக கொடுத்து டெம்பரேச்சரை பிரடிக்ட் செய்கிறோம்
            input_df = pd.DataFrame([{"CPU_Load": current_cpu_load}])
            predicted_temp = model.predict(input_df)[0]
            
            # 4. லைவ் ஸ்டேட்டஸ் பிரிண்ட் செய்கிறோம்
            print(f"💻 Real CPU Load : {current_cpu_load}%  |  🌡️ Predicted Temp : {predicted_temp:.2f}°C", end="  |  ")
            
            if predicted_temp < 70:
                print("🟢 Status: SAFE")
            elif 70 <= predicted_temp <= 82:
                print("🟡 Status: WARM")
            else:
                print("🔴 Status: HIGH TEMP ALERT!")
                
    except FileNotFoundError:
        print("❌ Error: 'cpu_temp_model.pkl' not found! Run Day 5 code first.")
    except KeyboardInterrupt:
        print("\n🛑 Live monitor stopped by user. Goodbye!")

if __name__ == "__main__":
    live_thermal_monitor()