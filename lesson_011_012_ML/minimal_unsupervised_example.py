import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fake data: two loose groups of points
rng = np.random.default_rng(42)
A = rng.normal(loc=[2, 2], scale=0.6, size=(40, 2))
B = rng.normal(loc=[6, 5], scale=0.6, size=(40, 2))
X = np.vstack([A, B])

# 1) Choose number of clusters k
kmeans = KMeans(n_clusters=2, n_init="auto", random_state=42)

# 2) Fit and get cluster labels (0 or 1)
labels = kmeans.fit_predict(X)

# 3) Inspect results
centers = kmeans.cluster_centers_
print("Cluster centers:\n", centers)
print("First 10 labels:", labels[:10])

# 4) Basic quality check (bigger silhouette ≈ better separation; roughly 0.5+ is decent)
print("Silhouette score:", silhouette_score(X, labels))

# TODO: Where is the code of plotting the centroids?!
