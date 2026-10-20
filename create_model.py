import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample Dataset (CPU Load vs Temp)
data = {
    "CPU_Load": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "Temp": [40, 45, 50, 56, 62, 68, 75, 82, 88, 95],
}
df = pd.DataFrame(data)

X = df[["CPU_Load"]]
y = df["Temp"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save .pkl File
joblib.dump(model, "cpu_temp_model.pkl")
print("✅ cpu_temp_model.pkl file successfully created!")