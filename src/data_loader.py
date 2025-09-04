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
