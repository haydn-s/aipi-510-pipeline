"""Data sourcing and ingestion for the wine classification pipeline."""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw():
    """Load the raw wine dataset as a DataFrame (features + 'target' column)."""
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    return df, list(data.target_names)


def summarize(df):
    """Print a quick sanity check (paste the output into the README)."""
    print(f"Rows: {df.shape[0]}, Features: {df.shape[1] - 1}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print("Class balance:")
    print(df["target"].value_counts().sort_index().to_string())


def load_data(test_size=0.2, random_state=42, scale=True):
    """Return X_train, X_test, y_train, y_test (scaled, stratified split)."""
    df, _ = load_raw()
    df = df.drop_duplicates().dropna()
    X, y = df.drop(columns="target"), df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:  # fit on train only to avoid leaking test info
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df, names = load_raw()
    print("Classes:", names)
    summarize(df)
    X_train, X_test, y_train, y_test = load_data()
    print("Train:", X_train.shape, "Test:", X_test.shape)
