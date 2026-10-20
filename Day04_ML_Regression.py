import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Dataset Generator
def prepare_dataset(n_samples=300):
    np.random.seed(42)
    cpu_load = np.random.uniform(20, 100, n_samples)
    cpu_temp = 35.0 + (0.55 * cpu_load) + np.random.normal(0, 3.5, n_samples)
    
    df = pd.DataFrame({"CPU_Load": cpu_load, "CPU_Temp": cpu_temp})
    return df

# 2. Machine Learning Pipeline
def train_and_evaluate():
    df = prepare_dataset()
    
    X = df[["CPU_Load"]]
    y = df["CPU_Temp"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("--- 🤖 DAY 4: MODEL TRAINING RESULTS ---")
    print(f"Model Formula Intercept (c): {model.intercept_:.2f}")
    print(f"Model Slope/Coefficient (m): {model.coef_[0]:.2f}")
    print(f"Mean Squared Error (MSE)   : {mse:.2f}")
    print(f"R2 Score (Accuracy Metric) : {r2 * 100:.2f}%\n")
    
    sample_load = np.array([[85.0]])
    predicted_temp = model.predict(sample_load)[0]
    print(f"🔮 PREDICTION: If CPU Load is 85%, Predicted Temp = {predicted_temp:.2f}°C")

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color="teal", label="Actual Test Data")
    plt.plot(X_test, y_pred, color="red", linewidth=2, label="AI Prediction Line")
    plt.title("Day 4: Linear Regression - CPU Load vs Temp Prediction")
    plt.xlabel("CPU Load (%)")
    plt.ylabel("CPU Temp (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()