"""Data preprocessing helpers for the ML project."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Split features/target and build a preprocessing pipeline.

    The target column is resolved using a small set of common names and
    falls back to the final column in the DataFrame when needed.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    target_candidates = ("target", "Target", "price", "Price")
    target_column = next((col for col in target_candidates if col in df.columns), df.columns[-1])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return X, y, preprocessor
