"""Data processing utilities for the auto insurance fraud project."""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load the insurance fraud CSV and clean structural/missing-value artifacts."""
    df = pd.read_csv(filepath)

    columns_to_delete = [
        "policy_number",
        "insured_zip",
        "insured_relationship",
        "incident_location",
        "auto_make",
        "_c39",
    ]
    df = df.drop(columns=[c for c in columns_to_delete if c in df.columns])

    if "authorities_contacted" in df.columns:
        df["authorities_contacted"] = df["authorities_contacted"].fillna("No Contact")

    columns_to_replace = [
        "property_damage",
        "collision_type",
        "police_report_available",
    ]
    existing = [c for c in columns_to_replace if c in df.columns]

    if existing:
        df[existing] = df[existing].replace("?", np.nan)
        for column in existing:
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    return df


def remove_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Remove observations with extreme numerical Z-scores.

    The binary target is excluded from outlier filtering.
    """
    df_clean = df.copy()

    numeric_columns = [
        c
        for c in df_clean.select_dtypes(include=[np.number]).columns
        if c != "fraud_reported"
    ]

    if not numeric_columns:
        return df_clean.reset_index(drop=True)

    mask = pd.Series(True, index=df_clean.index)

    for column in numeric_columns:
        values = df_clean[column].fillna(df_clean[column].median())
        scores = np.abs(zscore(values, nan_policy="omit"))
        column_mask = pd.Series(scores <= threshold, index=df_clean.index)
        mask &= column_mask.fillna(True)

    return df_clean.loc[mask].reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create policy duration and encode the fraud target."""
    df = df.copy()

    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
    df["incident_date"] = pd.to_datetime(df["incident_date"])

    df["Duration"] = (
        df["incident_date"] - df["policy_bind_date"]
    ).dt.days

    df = df.drop(columns=["policy_bind_date", "incident_date"])

    df["fraud_reported"] = df["fraud_reported"].replace({"Y": 1, "N": 0})

    nominal_columns = [
        "policy_state",
        "policy_csl",
        "insured_sex",
        "insured_occupation",
        "insured_hobbies",
        "incident_type",
        "collision_type",
        "authorities_contacted",
        "incident_state",
        "incident_city",
        "property_damage",
        "police_report_available",
        "auto_model",
    ]

    for column in nominal_columns:
        if column in df.columns:
            df[column] = df[column].astype(str)

    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    y = df["fraud_reported"]
    X = df.drop(columns=["fraud_reported"])

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
