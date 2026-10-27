"""
data_cleaning.py
----------------
Loads the loan approval dataset, handles missing values and outliers.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import mstats


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "loan_approval_dataset.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_train.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip whitespace from column names (dataset has leading spaces)
    df.columns = df.columns.str.strip()
    # Strip whitespace from string values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    print(f"✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("\n✅ No missing values found.")
        return pd.DataFrame()
    summary = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": (missing / len(df) * 100).round(2)
    })
    print("\n📋 Missing Value Summary:")
    print(summary)
    return summary


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include="object").columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"✅ Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def cap_outliers(df: pd.DataFrame, cols: list, limits=(0.01, 0.01)) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = mstats.winsorize(df[col], limits=list(limits))
    print(f"✅ Outliers capped for: {cols}")
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])
        print("🗑️ Dropped 'loan_id' column")
    return df


def clean_pipeline(save: bool = True) -> pd.DataFrame:
    df = load_data()
    report_missing(df)
    df = impute_missing(df)

    numeric_cols = ["income_annum", "loan_amount",
                    "residential_assets_value", "commercial_assets_value",
                    "luxury_assets_value", "bank_asset_value"]
    df = cap_outliers(df, numeric_cols)
    df = drop_unused(df)

    if save:
        df.to_csv(CLEANED_PATH, index=False)
        print(f"\n💾 Saved cleaned data to: {CLEANED_PATH}")

    return df


if __name__ == "__main__":
    clean_pipeline()