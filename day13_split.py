from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Sample Data (10 inputs)
X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

# Data-va Train (80%) and Test (20% - random-ah) split panrom
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train data:", X_train.ravel())
print("X_test data (Testingku mattum tharuvom):", X_test.ravel())

# Model Train panrom
model = LogisticRegression()
model.fit(X_train, y_train)

# Test data vechu check panrom
y_pred = model.predict(X_test)
print("Model Predictions for Test Data:", y_pred)
print("Actual Test Values:", y_test)