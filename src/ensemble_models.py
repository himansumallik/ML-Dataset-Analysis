import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Feature selection
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
y = pd.cut(df["Spending Score (1-100)"], bins=3, labels=["Low", "Medium", "High"])

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Ensemble Models ===

# 1️⃣ Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 2️⃣ AdaBoost
adb = AdaBoostClassifier(n_estimators=100, random_state=42)

# 3️⃣ Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 4️⃣ Voting Classifier (combines the above three)
voting_clf = VotingClassifier(estimators=[("rf", rf), ("adb", adb), ("gb", gb)], voting="soft")

# Train all models
models = {"Random Forest": rf, "AdaBoost": adb, "Gradient Boosting": gb, "Voting Ensemble": voting_clf}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

