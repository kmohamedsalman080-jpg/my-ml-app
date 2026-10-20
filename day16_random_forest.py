from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample Data: [CPU_Usage, Fan_Speed_Low]
X = np.array([
    [20, 0], 
    [30, 0], 
    [85, 1], 
    [90, 1], 
    [40, 0], 
    [95, 1]
])

# Target labels (0 = Safe, 1 = Overheat)
y = np.array([0, 0, 1, 1, 0, 1])

# Random Forest Model (100 Decision Trees sethu oru Forest create panrom)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model-a train panrom
model.fit(X, y)

# Pudhu data-voda test panrom (Eg: CPU 88%, Fan Low irukku)
test_data = np.array([[88, 1]])
prediction = model.predict(test_data)

print("Random Forest Prediction for High CPU & Low Fan:", prediction)