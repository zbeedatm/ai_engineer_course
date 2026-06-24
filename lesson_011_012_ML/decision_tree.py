import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# 1) Data from the table
X = np.array([
    [1,1],[1,0],[1,1],[1,0],
    [0,1],[0,1],[0,0],[0,0],[0,0],[0,0]
], dtype=int)
y = np.array([1,1,1,1, 1,0,0,0,0,0], dtype=int)

# 2) Train a small tree
clf = DecisionTreeClassifier(max_depth=2, criterion="gini", random_state=42)
clf.fit(X, y)

# 3) See the rules
print(export_text(clf, feature_names=["contains_free","many_exclamations"]))

# 4) Quick check (on this tiny set)
pred = clf.predict(X)
print("Accuracy:", accuracy_score(y, pred))

# 5) Try your own email
def predict_one(contains_free, many_exclamations):
    p = clf.predict_proba([[contains_free, many_exclamations]])[0,1]
    yhat = clf.predict([[contains_free, many_exclamations]])[0]
    print(f"Spam probability: {p:.3f}  → predicted: {yhat}")

predict_one(1, 0)  # expects spam
predict_one(0, 0)  # expects not spam
predict_one(0, 1)  # mixed branch
