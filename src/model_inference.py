"""Model inference helpers.

The functions here keep inference small and predictable: load the trained
pipeline, normalize incoming input into a DataFrame, and return a single
prediction with clear failures when the saved model or input shape is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a trained model pipeline from disk.

    Raises
    ------
    FileNotFoundError
        If the saved model does not exist.
    """

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Saved model not found at: {model_file}")

    return joblib.load(model_file)


def _get_expected_columns(model: Any) -> list[str]:
    """Infer the feature columns expected by the trained preprocessing step."""

    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return []


def _prepare_input_frame(input_dict: dict[str, Any], model: Any) -> pd.DataFrame:
    """Convert the incoming dictionary into a model-ready DataFrame.

    Extra keys are ignored and missing keys raise a helpful error so the UI can
    guide the user instead of failing with a cryptic scikit-learn traceback.
    """

    if not isinstance(input_dict, dict):
        raise TypeError("input_dict must be a dictionary of feature names to values.")

    expected_columns = _get_expected_columns(model)
    input_df = pd.DataFrame([input_dict])

    if expected_columns:
        missing_columns = [column for column in expected_columns if column not in input_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required input fields: {', '.join(missing_columns)}")

        # Keep the model input aligned to the training schema.
        input_df = input_df.reindex(columns=expected_columns)

    return input_df


def predict(input_dict: dict[str, Any], model_path: str | Path = DEFAULT_MODEL_PATH):
    """Generate a prediction for a single input record.

    Parameters
    ----------
    input_dict:
        A dictionary of feature names to values.
    model_path:
        Path to the saved joblib model pipeline.
    """
    model = load_model(model_path)
    input_df = _prepare_input_frame(input_dict, model)
    prediction = model.predict(input_df)
    # Return a plain float so API/UI layers can serialize cleanly.
    return float(prediction[0])
