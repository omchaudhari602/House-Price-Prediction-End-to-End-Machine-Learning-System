"""Model training entry point.

This module keeps the training workflow modular so it can be reused from the
CLI entry point, notebooks, or tests without duplicating data-loading logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_data, preprocess_data


TEST_SIZE = 0.2
RANDOM_STATE = 42


def _build_models() -> dict[str, Any]:
    """Create the candidate regression models."""

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _print_evaluation(model_name: str, y_true: Any, y_pred: Any) -> dict[str, float]:
    """Print regression metrics and return them for model selection."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"MAE:  {mae:,.4f}")
    print(f"RMSE: {rmse:,.4f}")
    print(f"R²:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def _train_and_select_best_model(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    preprocessor: Any,
) -> tuple[Pipeline, str]:
    """Fit each candidate model and return the best pipeline by RMSE."""

    best_model: Pipeline | None = None
    best_name = ""
    best_rmse = float("inf")
    best_r2 = float("-inf")

    for model_name, estimator in _build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = _print_evaluation(model_name, y_test, predictions)

        # Primary selection metric: RMSE (lower is better).
        # Tie-breaker: R² (higher is better).
        if metrics["rmse"] < best_rmse or (
            abs(metrics["rmse"] - best_rmse) < 1e-9 and metrics["r2"] > best_r2
        ):
            best_rmse = metrics["rmse"]
            best_r2 = metrics["r2"]
            best_model = pipeline
            best_name = model_name

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    return best_model, best_name


def _save_model(model: Pipeline, model_path: str | Path) -> Path:
    """Persist the trained pipeline to disk and return the output path."""

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def train_model_from_dataframe(df: Any, model_path: str | Path = "../models/model.pkl") -> Path:
    """Train and select the best regressor directly from a DataFrame."""

    X, y, preprocessor = preprocess_data(df)

    if not hasattr(y, "dtype"):
        raise ValueError("Target column is invalid.")

    if str(y.dtype) in {"object", "string"}:
        raise ValueError("Target column must be numeric for regression training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    best_model, best_name = _train_and_select_best_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    output_path = _save_model(best_model, model_path)

    print(f"\nBest Model: {best_name}")
    print(f"Saved to: {output_path.resolve()}")

    return output_path


def train_model(data_path: str, model_path: str = "../models/model.pkl") -> Path:
    """Load a CSV file, train and select the best regressor, then save it."""

    df = load_data(data_path)
    return train_model_from_dataframe(df, model_path=model_path)
