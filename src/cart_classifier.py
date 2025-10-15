import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features and target
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
y = pd.cut(df["Spending Score (1-100)"], bins=3, labels=["Low", "Medium", "High"])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train CART model
cart = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)
cart.fit(X_train, y_train)

# Predict and evaluate
y_pred = cart.predict(X_test)

print("\n=== CART Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(cart, feature_names=X.columns, class_names=cart.classes_, filled=True)
plt.title("CART Decision Tree for Customer Categorization")
plt.show()
