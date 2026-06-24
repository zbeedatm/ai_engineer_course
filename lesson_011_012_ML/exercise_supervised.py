import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1) Data (binary target): 0 = malignant, 1 = benign
data = load_breast_cancer()
X = data.data[:, :2]                 # use only first 2 features to keep it tiny
y = data.target                      # 0/1
feat_names = data.feature_names[:2]  # e.g., ["mean radius", "mean texture"]

# Print a tiny debug slice
print("Features used:", feat_names.tolist())
print("First 5 rows (x1, x2) with labels:")
for i in range(5):
    x1, x2 = X[i]
    print(f"  {x1:.2f}, {x2:.2f}  ->  {data.target_names[y[i]]}")

# 2) Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# 3) Quick (training-set) accuracy, like your minimal spam example
pred = clf.predict(X)
print("\nTraining accuracy:", round(accuracy_score(y, pred), 3))

# 4) Try your own values (interactive)
print(f"\nEnter two numbers for: {feat_names[0]} and {feat_names[1]}. Empty input to quit.")
while True:
    s1 = input(f"{feat_names[0]}: ").strip()
    if s1 == "":
        print("Bye.")
        break
    s2 = input(f"{feat_names[1]}: ").strip()
    if s2 == "":
        print("Bye.")
        break
    try:
        row = np.array([[float(s1), float(s2)]], dtype=float)
    except ValueError:
        print("Invalid numbers, try again.\n")
        continue

    proba = clf.predict_proba(row)[0]           # [P(class 0), P(class 1)]
    label = clf.predict(row)[0]                 # 0 or 1
    # Map probabilities to class names robustly
    for cls_idx, p in zip(clf.classes_, proba):
        print(f"  P({data.target_names[cls_idx]}): {p:.3f}")
    print("  Predicted class:", data.target_names[label], "\n")
