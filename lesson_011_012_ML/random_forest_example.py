# random_forest_tiny.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1) Data from the table
X = np.array([
    [1,1],
    [1,0],
    [1,1],
    [0,1],
    [0,1],  # noisy row: says exclamations but not spam
    [0,0],
    [0,0],
    [1,0],
    [0,0],
    [1,1],
], dtype=int)

y = np.array([1,1,1,1,0,0,0,1,0,1], dtype=int)  # spam labels

# 2) Random Forest (small but sturdy)
rf = RandomForestClassifier(
    n_estimators=50,      # number of trees (keep small)
    max_depth=None,       # let trees grow; forest averages them
    random_state=42
)
rf.fit(X, y)

# 3) Quick check on this tiny training set (demo only)
pred = rf.predict(X)
print("Accuracy on this tiny table:", round(accuracy_score(y, pred), 3))

# 4) Which feature matters more?
print("Feature importances [contains_free, many_exclamations]:",
      np.round(rf.feature_importances_, 3))

# 5) Try your own email (enter 0/1)
def predict_one(contains_free, many_exclamations, threshold=0.5):
    row = np.array([[contains_free, many_exclamations]], dtype=int)
    p = rf.predict_proba(row)[0,1]
    yhat = int(p >= threshold)
    print(f"Spam probability: {p:.3f}  → predicted: {yhat}")

# Examples (change freely)
predict_one(1, 1)  # likely spam
predict_one(1, 0)  # likely spam
predict_one(0, 1)  # depends (noisy), forest will vote
predict_one(0, 0)  # likely not spam
