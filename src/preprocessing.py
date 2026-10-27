"""
preprocessing.py
----------------
Feature engineering, encoding, and scaling.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_train.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def load_cleaned(path: str = CLEANED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Loaded cleaned data: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing asset and income data."""
    df = df.copy()

    # Total asset value (residential + commercial + luxury + bank)
    df["total_assets"] = (df["residential_assets_value"] +
                          df["commercial_assets_value"] +
                          df["luxury_assets_value"] +
                          df["bank_asset_value"])

    # Loan-to-income ratio
    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1)

    # Loan-to-assets ratio
    df["loan_asset_ratio"] = df["loan_amount"] / (df["total_assets"] + 1)

    # Monthly loan payment estimate (simple: loan_amount / loan_term)
    df["monthly_payment"] = df["loan_amount"] / df["loan_term"]

    # Disposable income after monthly payment
    df["disposable_income"] = df["income_annum"] - df["monthly_payment"]

    # Log transform skewed numerical features
    for col in ["income_annum", "loan_amount", "total_assets"]:
        df[col] = np.log1p(df[col])

    print("✅ Engineered features: total_assets, loan_income_ratio, "
          "loan_asset_ratio, monthly_payment, disposable_income")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "education" in df.columns:
        df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})

    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})

    print(f"✅ Encoded categoricals. Shape now: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame):
    if "loan_status" not in df.columns:
        raise ValueError("loan_status column missing!")
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    print(f"✅ Features: {X.shape[1]}, Target distribution:\n{y.value_counts()}")
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Saved scaler to: {SCALER_PATH}")
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(save: bool = True):
    from sklearn.model_selection import train_test_split

    df = load_cleaned()
    df = engineer_features(df)
    df = encode_categoricals(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test, _ = scale_features(X_train, X_test)

    if save:
        X_train.to_csv(os.path.join(DATA_DIR, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_DIR, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)
        print("💾 Saved train/test splits.")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_pipeline()