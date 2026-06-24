import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# features and labels
X = np.array([
    [1,3],
    [0,0],
    [1,2],
    [0,0],
    [0,1],
    [0,0]
], dtype=float)

y = np.array([1,0,1,0,1,0])  # spam? 1=yes, 0=no

# 1) choose a model
clf = LogisticRegression()

# 2) learn from labeled examples
clf.fit(X, y)

# 3) predict a new email
new_email = np.array([[1,1]])  # has "free", one "!"
print("Spam probability:", clf.predict_proba(new_email)[0,1])

# 4) evaluate on the tiny training set (for demo only)
pred = clf.predict(X)
print("Accuracy:", accuracy_score(y, pred))