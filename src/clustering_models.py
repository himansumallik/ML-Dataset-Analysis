# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load Iris dataset
# iris = load_iris()
# X = iris.data  # features

# # Apply K-Means
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(X)
# labels = kmeans.labels_

# # Add cluster labels to DataFrame
# df = pd.DataFrame(X, columns=iris.feature_names)
# df['Cluster'] = labels

# # Show first 5 rows
# print(df.head())

# # Plot clusters (first two features)
# plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.title("K-Means Clustering - Iris Dataset")
# plt.show()

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data  # all features

# Apply Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Add cluster labels to DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = labels

# Show first 5 rows
print(df.head())

# Plot clusters (first two features)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("GMM Clustering - Iris Dataset")
plt.show()
