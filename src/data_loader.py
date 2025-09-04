import os
import pandas as pd

# Path to your dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "paysim.csv")

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Show first 10 rows
print("First 5 rows of the dataset:\n")
print(df.head(5))

# Show column names
print("\nColumns in the dataset:\n")
print(df.columns.tolist())

# Show basic info
print("\nDataset info:\n")
print(df.info())


# 1️⃣ Shape of dataset
print("Shape of dataset:", df.shape, "\n")  # rows x columns

# 4️⃣ Summary statistics for numeric columns
print("Summary statistics (numeric columns):\n")
print(df.describe(), "\n")


# 5️⃣ Summary statistics for all columns including categorical
print("Summary statistics (all columns):\n")
print(df.describe(include='all'), "\n")

# 6️⃣ Check for missing values
print("Missing values per column:\n")
print(df.isnull().sum(), "\n")

# 7️⃣ Class distribution for target (isFraud)
if 'isFraud' in df.columns:
    print("Fraud class distribution:\n")
    print(df['isFraud'].value_counts(), "\n")
    print("Percentage of fraud vs non-fraud:\n")
    print(df['isFraud'].value_counts(normalize=True) * 100)
