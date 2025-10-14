import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------- K-MEANS CLUSTERING (Iris Dataset) --------------------
print("\n=== K-Means Clustering on Iris Dataset ===")
iris = load_iris()
X_iris = iris.data

kmeans = KMeans(n_clusters=3, random_state=42)
iris_labels = kmeans.fit_predict(X_iris)

df_iris = pd.DataFrame(X_iris, columns=iris.feature_names)
df_iris["Cluster"] = iris_labels
print(df_iris.head())

plt.scatter(df_iris.iloc[:, 0], df_iris.iloc[:, 1], c=df_iris["Cluster"], cmap="viridis")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("K-Means Clustering - Iris Dataset")
plt.show()

# -------------------- GAUSSIAN MIXTURE MODEL (Mall Customers) --------------------
print("\n=== Gaussian Mixture Model on Mall Customers Dataset ===")
df_mall = pd.read_csv("data/Mall_Customers.csv")
X_mall = df_mall[["Annual Income (k$)", "Spending Score (1-100)"]]

gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(X_mall)

plt.scatter(X_mall["Annual Income (k$)"], X_mall["Spending Score (1-100)"], c=gmm_labels, cmap="viridis")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Gaussian Mixture Clustering - Mall Customers")
plt.show()

# -------------------- HIERARCHICAL CLUSTERING (Mall Customers) --------------------
print("\n=== Hierarchical Clustering on Mall Customers Dataset ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_mall)

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(linked, labels=df_mall["CustomerID"].values, orientation="top", distance_sort="descending", show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram - Mall Customers")
plt.xlabel("Customer ID")
plt.ylabel("Distance")
plt.show()
