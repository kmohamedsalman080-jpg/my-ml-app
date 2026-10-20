from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Sample Data: [CPU_Usage, Fan_Speed_Low]
X = np.array([
    [20, 0], [30, 0], [85, 1], [90, 1], 
    [40, 0], [95, 1], [70, 1], [15, 0]
])
y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

# Try panna vendiya Hyperparameters (Grid Options)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [2, 4, None]
}

# Base Model
rf = RandomForestClassifier(random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, scoring='accuracy')
grid_search.fit(X, y)

print("--- Best Hyperparameters Found ---")
print("Best Params:", grid_search.best_params_)
print(f"Best Accuracy: {grid_search.best_score_ * 100:.2f}%")