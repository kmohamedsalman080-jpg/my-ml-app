import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib  # Model-ah save panna indha library dhaan use agum

# 1. Model Train panra steps (Short-a oru model ready panrom)
def train_and_save_model():
    # Dummy CPU load and temp data
    np.random.seed(42)
    cpu_load = np.random.uniform(20, 100, 300)
    cpu_temp = 35.0 + (0.55 * cpu_load) + np.random.normal(0, 3.5, 300)
    
    X = pd.DataFrame({"CPU_Load": cpu_load})
    y = cpu_temp
    
    # Model train panrom
    model = LinearRegression()
    model.fit(X, y)
    
    # 2. Model-ah 'cpu_temp_model.pkl' nu save panrom
    joblib.dump(model, "cpu_temp_model.pkl")
    print("✅ Success: AI Model trained and saved as 'cpu_temp_model.pkl'!")

# 3. Save aana model-ah load panni real-time prediction panrathu
def load_and_predict(new_cpu_load):
    # Model file-ah load panrom
    loaded_model = joblib.load("cpu_temp_model.pkl")
    
    # Input-ah kuduthu temperature predict panrom
    prediction = loaded_model.predict(np.array([[new_cpu_load]]))
    return prediction[0]

if __name__ == "__main__":
    # First model-ah train panni save panrom
    train_and_save_model()
    
    # Ippo app or software-la irukkaru mathiri real-time values check panrom
    current_loads = [45.0, 75.0, 90.0]
    print("\n--- 🌡️ REAL-TIME CPU TEMPERATURE PREDICTIONS ---")
    for load in current_loads:
        predicted_temp = load_and_predict(load)
        print(f"CPU Load: {load}% ---> Predicted Temperature: {predicted_temp:.2f}°C")