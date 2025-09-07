import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/paysim.csv")

# Features and target
X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df['isFraud']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = StandardScaler().fit_transform(X_train), StandardScaler().fit_transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("\nLogistic Regression:\n", classification_report(y_test, y_pred_log))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression"); plt.show()

# SVM
svm = SVC(kernel="linear").fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSVM:\n", classification_report(y_test, y_pred_svm))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Greens")
plt.title("SVM"); plt.show()
