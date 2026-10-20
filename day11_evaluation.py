from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Real / Actual system states (0 = Safe, 1 = Overheat)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])

# Model predictions (Enna output sonnuchu)
y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# 2. Accuracy Score
acc = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {acc * 100}%")