"""Streamlit app for model inference.

The UI is generated from the dataset schema so the form adapts automatically
when columns change. This keeps the app simple to maintain and reduces the
chance of feature mismatches at prediction time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.model_inference import DEFAULT_MODEL_PATH, predict
from src.model_training import train_model_from_dataframe


st.set_page_config(page_title="ML Project Predictor", page_icon="🏠", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TARGET_CANDIDATES = ("target", "Target", "price", "Price")


def find_dataset_path() -> Path:
    """Find the first CSV dataset in the project data folder."""
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV dataset found in {DATA_DIR}.")
    return csv_files[0]


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the dataset used to generate the inference form."""

    dataset_path = find_dataset_path()
    return pd.read_csv(dataset_path)


def get_target_column(df: pd.DataFrame) -> str:
    """Resolve the target column from known names or fall back to the last column."""
    return next((col for col in TARGET_CANDIDATES if col in df.columns), df.columns[-1])


def build_input_fields(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Generate Streamlit widgets for every feature column."""
    feature_columns = [col for col in df.columns if col != target_column]
    sample_row = df.iloc[0]
    input_data: dict[str, Any] = {}

    st.subheader("Enter feature values")
    with st.form("prediction_form"):
        for column in feature_columns:
            series = df[column]
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_numeric:
                default_value = float(sample_row[column])
                if pd.api.types.is_integer_dtype(series):
                    input_data[column] = st.number_input(
                        label=column,
                        value=int(default_value),
                        step=1,
                    )
                else:
                    input_data[column] = st.number_input(
                        label=column,
                        value=default_value,
                    )
            else:
                unique_values = series.dropna().astype(str).unique().tolist()
                default_value = str(sample_row[column])
                if 0 < len(unique_values) <= 10:
                    input_data[column] = st.selectbox(
                        label=column,
                        options=unique_values,
                        index=unique_values.index(default_value) if default_value in unique_values else 0,
                    )
                else:
                    input_data[column] = st.text_input(label=column, value=default_value)

        submitted = st.form_submit_button("Predict")

    return input_data if submitted else {}


def main() -> None:
    """Render the Streamlit prediction app."""
    st.title("🏠 Machine Learning Predictor")
    st.write("Automatically generated inputs based on the dataset columns.")

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.error("The dataset is empty.")
        return

    target_column = get_target_column(df)
    feature_columns = [col for col in df.columns if col != target_column]

    with st.expander("Dataset preview", expanded=False):
        st.dataframe(df.head())
        st.caption(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    st.markdown("---")
    st.subheader("Feature summary")
    st.write(f"Target column: `{target_column}`")
    st.write(f"Feature columns: {', '.join(feature_columns)}")

    input_data = build_input_fields(df, target_column)
    if not input_data:
        return

    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        st.error(f"Saved model not found at: {model_path}")
        return

    try:
        prediction = predict(input_data, model_path=model_path)
    except AttributeError as exc:
        # Older serialized sklearn objects can break after package upgrades.
        # Rebuild the model once from the current dataset and try again.
        if "SimpleImputer" in str(exc) or "_fill_dtype" in str(exc):
            st.warning("Saved model appears incompatible with the current environment. Rebuilding it now...")
            try:
                train_model_from_dataframe(df, model_path=model_path)
                prediction = predict(input_data, model_path=model_path)
            except Exception as rebuild_exc:  # pragma: no cover - surfaced in UI
                st.error(f"Prediction failed after rebuilding the model: {rebuild_exc}")
                return
        else:
            st.error(f"Prediction failed: {exc}")
            return
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Prediction failed: {exc}")
        return
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(f"An unexpected error occurred during prediction: {exc}")
        return

    st.success("Prediction completed successfully.")
    st.metric("Predicted Price", f"{prediction:,.2f}")
    st.json({"input": input_data, "prediction": prediction})


if __name__ == "__main__":
    main()
