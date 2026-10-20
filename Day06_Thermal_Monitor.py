import joblib
import pandas as pd

def run_thermal_monitor():
    try:
        # Save பண்ணின மாடலை லோட் பண்றோம்
        model = joblib.load("cpu_temp_model.pkl")
        print("==========================================")
        print("  💻 SMART CPU THERMAL MONITOR SYSTEM  ")
        print("==========================================")
        
        while True:
            user_input = input("\nEnter current CPU Load % (or type 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                print("Exiting Thermal Monitor. Goodbye!")
                break
                
            try:
                cpu_load = float(user_input)
                
                if cpu_load < 0 or cpu_load > 100:
                    print("⚠️ Please enter a valid percentage between 0 and 100.")
                    continue
                
                # Feature name warning வராம இருக்க Pandas DataFrame-ஆ கொடுக்கிறோம்
                input_df = pd.DataFrame([{"CPU_Load": cpu_load}])
                predicted_temp = model.predict(input_df)[0]
                
                print(f"\n📊 Input CPU Load : {cpu_load}%")
                print(f"🌡️ Predicted Temp : {predicted_temp:.2f}°C")
                
                # Status Alert System
                if predicted_temp < 70:
                    print("🟢 Status: COOL & SAFE (Normal Operations)")
                elif 70 <= predicted_temp <= 82:
                    print("🟡 Status: WARM (Heavy Tasks/Multitasking)")
                else:
                    print("🔴 Status: HIGH TEMP ALERT! (Cooling Fan Speed Max Required)")
                    
            except ValueError:
                print("❌ Invalid input! Please enter a numerical value.")

    except FileNotFoundError:
        print("❌ Error: 'cpu_temp_model.pkl' file not found! Run Day 5 code first.")

if __name__ == "__main__":
    run_thermal_monitor()