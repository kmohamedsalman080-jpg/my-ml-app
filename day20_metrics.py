from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Actual target values (True labels)
y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])

# Model predict panna values (Predicted labels)
y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])  # Notice: Oru case-a miss pannirukku (False Negative)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("--- Confusion Matrix ---")
print(cm)

print("\n--- Detailed Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Safe (0)', 'Overheat (1)']))