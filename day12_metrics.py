from sklearn.metrics import classification_report, precision_score, recall_score
import numpy as np

# Real / Actual system states (0 = Safe, 1 = Overheat)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])

# Model predictions
y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 1])

# 1. Individual Scores
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# 2. Complete Report (Ellathaiyum oru adiyila tharum!)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))