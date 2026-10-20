import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. CPU Load (%)
X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]])

# 2. Target: 0 = Safe (<70% load), 1 = Overheat (>=70% load)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

# 3. Model Create & Train
model = LogisticRegression()
model.fit(X, y)

# 4. Predict for new CPU Load
test_load = [[75]]
prediction = model.predict(test_load)
probability = model.predict_proba(test_load)

status = "🔥 Overheating/Critical" if prediction[0] == 1 else "✅ Safe Operation"

print(f"CPU Load: {test_load[0][0]}%")
print(f"Prediction Status: {status}")
print(f"Probability (Safe vs Critical): {probability[0]}")