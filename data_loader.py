import pandas as pd
from typing import Tuple

TELCO_TARGET = "Churn"
TELCO_ID = "customerID"


def load_csv(file_or_path) -> pd.DataFrame:
    """Load CSV from uploaded file or path."""
    df = pd.read_csv(file_or_path)
    return df


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning tailored to Telco dataset.
    - Coerce TotalCharges to numeric
    - Drop rows with missing TotalCharges
    - Strip whitespace in string columns
    - Normalize target to 0/1
    """
    df = df.copy()

    # Coerce TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])  # small number of rows

    # Strip whitespace
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Standardize target
    if TELCO_TARGET in df.columns:
        df[TELCO_TARGET] = df[TELCO_TARGET].map({"Yes": 1, "No": 0}).astype(int)

    return df


def split_X_y(df: pd.DataFrame, target: str = TELCO_TARGET, drop_ids: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    if drop_ids and TELCO_ID in X.columns:
        X = X.drop(columns=[TELCO_ID])
    y = df[target]
    return X, y