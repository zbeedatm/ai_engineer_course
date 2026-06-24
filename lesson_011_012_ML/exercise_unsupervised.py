import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 0) Load a real dataset (Iris)
data = load_iris()
# Keep it tiny/transparent: use 2 features (sepal length, petal length)
X = data.data[:, [0, 2]]
feat_names = [data.feature_names[0], data.feature_names[2]]

# 1) Choose number of clusters k (Iris has 3 species)
kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42)

# 2) Fit and get cluster labels
labels = kmeans.fit_predict(X)

# 3) Inspect results
centers = kmeans.cluster_centers_
print("Using features:", feat_names)
print("Cluster centers:\n", centers)
print("First 10 labels:", labels[:10])

# 4) Basic quality check (higher silhouette ≈ better separation)
print("Silhouette score:", silhouette_score(X, labels))

# (Optional) quick sanity check vs. true species (not used in training!)
df = pd.DataFrame({
    "cluster": labels,
    "true_species": [data.target_names[i] for i in data.target]
})
print("\nCluster vs true species (sanity only):")
print(pd.crosstab(df["cluster"], df["true_species"]))

# 5) Plot the 2D distribution (colored by cluster) + centroids
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)                 # points by cluster
plt.scatter(centers[:, 0], centers[:, 1], s=200, marker="X", edgecolor="k")  # centroids
for cid, (sx, px) in enumerate(centers):
    plt.text(sx + 0.05, px + 0.05, f"C{cid}", fontsize=10)    # label centroids

plt.xlabel(feat_names[0])                                     # sepal length (cm)
plt.ylabel(feat_names[1])                                     # petal length (cm)
plt.title("Iris • K-Means on [sepal length, petal length]")
plt.tight_layout()
plt.show()
