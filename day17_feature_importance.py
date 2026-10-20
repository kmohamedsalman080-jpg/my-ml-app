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

# Model Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature Importances extract panrom
importances = model.feature_importances_
feature_names = ['CPU Usage', 'Fan Speed Low']

print("--- Feature Importance Results ---")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance * 100:.2f}%")