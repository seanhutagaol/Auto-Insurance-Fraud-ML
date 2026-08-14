"""Data processing utilities for the auto insurance fraud project."""
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    drop_cols = ["policy_number","insured_zip","insured_relationship",
                 "incident_location","auto_make","_c39"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if "authorities_contacted" in df.columns:
        df["authorities_contacted"] = df["authorities_contacted"].fillna("No Contact")
    replace_cols = [c for c in ["property_damage","collision_type",
                                "police_report_available"] if c in df.columns]
    if replace_cols:
        df[replace_cols] = df[replace_cols].replace("?", np.nan)
        for c in replace_cols:
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
    return df

def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=[np.number]).columns:
        if c == "fraud_reported":
            continue
        vals = out[c].fillna(out[c].median())
        z = np.abs(zscore(vals, nan_policy="omit"))
        out = out.loc[(z <= threshold) | out[c].isna()].copy()
    return out

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "policy_bind_date" in df.columns and "incident_date" in df.columns:
        df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
        df["incident_date"] = pd.to_datetime(df["incident_date"])
        df["Duration"] = (df["incident_date"] - df["policy_bind_date"]).dt.days
        df = df.drop(columns=["policy_bind_date","incident_date"])
    if "fraud_reported" in df.columns:
        df["fraud_reported"] = df["fraud_reported"].replace({"Y":1,"N":0})
    nominal = ["policy_state","policy_csl","insured_sex","insured_occupation",
               "insured_hobbies","incident_type","collision_type",
               "authorities_contacted","incident_state","incident_city",
               "property_damage","police_report_available","auto_model"]
    for c in nominal:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def get_train_test_split(df: pd.DataFrame, test_size: float = 0.20,
                         random_state: int = 42) -> Tuple:
    y = df["fraud_reported"]
    X = df.drop(columns=["fraud_reported"])
    return train_test_split(X, y, test_size=test_size,
                             random_state=random_state, stratify=y)
