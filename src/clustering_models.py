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
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Load Mall Customers dataset
df = pd.read_csv("data/Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply GMM
gmm = GaussianMixture(n_components=5, random_state=42)
labels = gmm.fit_predict(X)

# Plot clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=labels, cmap='viridis')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Gaussian Mixture Clustering - Mall Customers")
plt.show()
