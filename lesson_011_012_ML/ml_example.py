# -----------------------------------------------
# Loan Decision Tree Visualization with scikit-learn
# -----------------------------------------------
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# Features: [monthly_income, credit_score]
X = np.array([
    [2000, 550],   # low income, poor credit
    [3000, 600],   # low income, medium credit
    [7000, 720],   # high income, good credit
    [8000, 680],   # high income, medium credit
    [4000, 750],   # medium income, good credit
])

# Labels: 0 = reject, 1 = approve
y = np.array([0, 0, 1, 1, 1])


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

plt.figure(figsize=(10, 6))
tree.plot_tree(
    clf,
    feature_names=["Monthly Income", "Credit Score"],
    class_names=["Reject", "Approve"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Loan Decision Tree (scikit-learn)")
plt.show()