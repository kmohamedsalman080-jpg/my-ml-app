from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import joblib

# 1. Sample Data & Training (Gradient Boosting Model)
X = np.array([
    [20, 0], [30, 0], [85, 1], [90, 1], 
    [40, 0], [95, 1], [70, 1], [15, 0]
])
y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# 2. Model-a File-ah Save panrom (Joblib use panni)
joblib.dump(model, 'server_model.pkl')
print("Model successfully saved as 'server_model.pkl'!")

# 3. Save pannatha thirumba Load panrom (Testing loading)
loaded_model = joblib.load('server_model.pkl')

# 4. Loaded model-a vachi test panrom
test_data = np.array([[88, 1]])
prediction = loaded_model.predict(test_data)
print("Prediction from Loaded Model for High CPU & Low Fan:", prediction)