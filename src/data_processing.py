"""Data Processing Module.

This module handles data loading, missing value imputation, outlier 
filtration using Z-scores, and temporal feature engineering for the 
auto insurance fraud dataset.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load the insurance dataset from an Excel file and clean structural artifacts.

    Parameters
    ----------
    filepath : str
        The path to the Excel file containing the dataset.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with dropped irrelevant columns and imputed values.
    """
    df = pd.read_csv(filepath)

    # Drop structural identifiers and artifact columns
    columns_to_delete = [
        'policy_number',
        'insured_zip',
        'insured_relationship',
        'incident_location',
        'auto_make',
        '_c39'
    ]
    df = df.drop(columns=[col for col in columns_to_delete if col in df.columns])

    # Impute missing strings/categories
    df['authorities_contacted'] = df['authorities_contacted'].fillna("No Contact")

    columns_to_replace = ["property_damage", "collision_type", "police_report_available"]
    df[columns_to_replace] = df[columns_to_replace].replace("?", np.nan)

    for column in columns_to_replace:
        mode_value = df[column].mode()[0]
        df[column] = df[column].fillna(mode_value)

    return df


def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Filter out numerical outliers based on a Z-score threshold.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    threshold : float, default=3.0
        The absolute Z-score threshold beyond which data points are considered outliers.

    Returns
    -------
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    df_clean = df.copy()
    
    for column in df_clean.select_dtypes(include=[float, int]).columns:
        # Compute absolute z-scores, filling NaNs temporarily to protect missing data rows
        col_data = df_clean[column].fillna(df_clean[column].median())
        z_scores = np.abs(zscore(col_data))
        df_clean = df_clean[(z_scores <= threshold) | (df_clean[column].isna())]

    return df_clean


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering, calculate duration intervals, and encode targets.

    Parameters
    ----------
    df : pd.DataFrame
        The pre-cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame ready for modeling.
    """
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
    df['incident_date'] = pd.to_datetime(df['incident_date'])

    # Calculate temporal duration between policy bind and incident
    df['Duration'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df = df.drop(columns=['policy_bind_date', 'incident_date'])

    # Encode binary target variable
    df['fraud_reported'] = df['fraud_reported'].replace({'Y': 1, 'N': 0})

    # Enforce string types for nominal categories
    nominal_columns = [
        'policy_state', 'policy_csl', 'insured_sex', 'insured_occupation', 
        'insured_hobbies', 'incident_type', 'collision_type', 'authorities_contacted', 
        'incident_state', 'incident_city', 'property_damage', 'police_report_available', 
        'auto_model'
    ]
    for col in nominal_columns:
        df[col] = df[col].astype(str)

    return df


def get_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into stratified training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        The fully engineered DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test splits.
    """
    y = df['fraud_reported']
    X = df.drop(['fraud_reported'], axis=1)

    return train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
