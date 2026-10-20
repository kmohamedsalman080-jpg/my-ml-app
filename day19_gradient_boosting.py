from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Sample Data: [CPU_Usage, Fan_Speed_Low]
X = np.array([
    [20, 0], [30, 0], [85, 1], [90, 1], 
    [40, 0], [95, 1], [70, 1], [15, 0]
])
y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

# Gradient Boosting Model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Model-a train panrom
model.fit(X, y)

# Pudhu data-voda test panrom (Eg: CPU 88%, Fan Low irukku)
test_data = np.array([[88, 1]])
prediction = model.predict(test_data)

print("Gradient Boosting Prediction for High CPU & Low Fan:", prediction)