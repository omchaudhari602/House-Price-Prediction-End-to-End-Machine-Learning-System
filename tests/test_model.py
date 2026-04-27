"""Tests for model training and inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model_inference import predict
from src.model_training import train_model_from_dataframe


def _load_training_dataframe() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "house_prices (1).csv"
    return pd.read_csv(data_path)


def test_train_model_creates_artifact(tmp_path):
    df = _load_training_dataframe()
    model_path = tmp_path / "model.pkl"

    output_path = train_model_from_dataframe(df, model_path=model_path)

    assert output_path.exists()
    assert output_path.suffix == ".pkl"


def test_predict_returns_numeric_value(tmp_path):
    df = _load_training_dataframe()
    model_path = tmp_path / "model.pkl"
    train_model_from_dataframe(df, model_path=model_path)

    sample_input = {
        "Property_ID": "PROP0001",
        "Area": 3712,
        "Bedrooms": 4,
        "Bathrooms": 3,
        "Age": 36,
        "Location": "Rural",
        "Property_Type": "House",
    }

    prediction = predict(sample_input, model_path=model_path)

    assert isinstance(prediction, float)
