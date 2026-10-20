from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample Data (10 inputs)
X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

# Logistic Regression Model
model = LogisticRegression()

# 5-Fold Cross Validation (cv=5)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("Ovvoru fold-oda accuracy scores:", scores)
print(f"Average Accuracy: {scores.mean() * 100}%")