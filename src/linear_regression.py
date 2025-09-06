import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Path to dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "paysim.csv")

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("✅ Dataset loaded for Linear Regression\n")

# --- Step 1: Select features and target ---
X = df[['oldbalanceOrg', 'amount']]  # Features
y = df['newbalanceOrig']             # Target

# --- Step 2: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Train linear regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 4: Predictions ---
y_pred = model.predict(X_test)

# --- Step 5: Evaluation ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# --- Step 6: Graph (Actual vs Predicted) ---
plt.figure(figsize=(8,6))
plt.scatter(y_test[:1000], y_pred[:1000], alpha=0.5, color="blue", label="Predicted vs Actual")  
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual newbalanceOrig")
plt.ylabel("Predicted newbalanceOrig")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.show()
