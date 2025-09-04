import os
import pandas as pd

# Path to your dataset folder
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "UCI_HAR_Dataset")


def load_features():
    """Load feature names and ensure they are unique"""
    features_path = os.path.join(DATASET_PATH, "features.txt")

    # Read both columns: index and name
    features = pd.read_csv(
        features_path,
        sep=r"\s+",
        header=None,
        names=["Index", "Feature"]
    )["Feature"]

    # Convert to string and make unique if duplicates exist
    features = features.astype(str)
    if features.duplicated().any():
        features = features + "_" + features.groupby(features).cumcount().astype(str)

    return features.tolist()



def load_activity_labels():
    """Load mapping of activity ID to activity name"""
    labels_path = os.path.join(DATASET_PATH, "activity_labels.txt")

    # Explicit column names
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["ActivityID", "ActivityName"]
    )

    return labels.set_index("ActivityID")["ActivityName"].to_dict()



def load_split(split, feature_names):
    """Load train/test split"""
    split_path = os.path.join(DATASET_PATH, split)

    # Features
    x_path = os.path.join(split_path, f"X_{split}.txt")
    X = pd.read_csv(x_path, sep=r"\s+", header=None, names=feature_names)

    # Labels
    y_path = os.path.join(split_path, f"y_{split}.txt")
    y = pd.read_csv(y_path, sep=r"\s+", header=None, names=["Activity"])

    # Subject IDs
    subject_path = os.path.join(split_path, f"subject_{split}.txt")
    subject = pd.read_csv(subject_path, sep=r"\s+", header=None, names=["Subject"])

    # Combine into single dataframe
    df = pd.concat([subject, y, X], axis=1)
    return df


def load_all():
    """Load full dataset (train + test)"""
    feature_names = load_features()
    activity_map = load_activity_labels()

    train = load_split("train", feature_names)
    test = load_split("test", feature_names)

    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    return df, activity_map


if __name__ == "__main__":
    df, activity_map = load_all()

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nActivity mapping:", activity_map)
