"""Main entry point for dataset inspection and model training.

This script is intentionally simple: it loads the dataset once, prints a quick
EDA summary, prepares preprocessing, and trains the models. Keeping the logic
here lightweight makes it easier to run end-to-end without ceremony.
"""

from __future__ import annotations

from pathlib import Path

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model_from_dataframe


def _resolve_data_path() -> Path:
    """Resolve the dataset path used by the training script."""

    return Path(__file__).resolve().parent / "data" / "house_prices (1).csv"


def main() -> None:
    """Load the dataset, run basic EDA, preprocess data, and train models."""

    project_root = Path(__file__).resolve().parent
    data_path = _resolve_data_path()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = load_data(str(data_path))

    print("=== First Few Rows ===")
    print(df.head())

    print("\n=== Dataset Info ===")
    df.info()

    print("\n=== Summary Statistics ===")
    print(df.describe(include="all"))

    X, y, preprocessor = preprocess_data(df)
    print(f"\nPreprocessing ready: {len(X.columns)} feature columns, target='{y.name}'")
    print(f"Preprocessor: {preprocessor.__class__.__name__}")

    model_path = project_root / "models" / "model.pkl"
    train_model_from_dataframe(df, model_path=str(model_path))


if __name__ == "__main__":
    main()
