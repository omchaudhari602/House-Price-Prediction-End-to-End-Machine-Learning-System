# Open Documentation

This file consolidates all previous markdown documentation from the `docs/` folder into a single source.

---

## Executive Summary

### Project Title

House Price Prediction — End-to-End Machine Learning System

### Purpose

This project provides a complete machine learning workflow for house price prediction using structured tabular data. It demonstrates practical ML engineering fundamentals by covering data preprocessing, model training, evaluation, persistence, and serving.

### Business Value

- Enables rapid price estimation from key property attributes.
- Provides two consumption modes:
  - interactive use through Streamlit
  - system integration through Flask API
- Reduces manual valuation effort and accelerates exploratory pricing analysis.

### Technical Highlights

- Unified preprocessing and model pipeline with scikit-learn
- Automatic target resolution and schema-aware inference validation
- Multi-model regression training with metric-based selection
- Persisted artifact for consistent, repeatable predictions
- Basic automated tests for training and inference behavior

### Quality and Reliability

- Modular source layout under `src/`
- Structured docs under `docs/`
- Test coverage for core training and prediction paths
- Health endpoint for operational checks

### Current Status

- Core pipeline implemented and functional
- Streamlit and Flask interfaces available
- Documentation upgraded to professional standard
- Additional screenshot (`api_response.png`) still recommended for complete visual evidence set

### Next Priorities

1. Expand test coverage (API tests and failure-mode tests)
2. Add CI workflow for automated quality checks
3. Introduce model lifecycle features (versioning/monitoring)
4. Add containerized deployment profile

---

## Project Overview

### What This Project Solves

This repository implements an end-to-end regression system for house price prediction, from data preparation through model serving.

### Key Capabilities

- Automated preprocessing for mixed numeric/categorical features
- Multi-model training with metric-driven best-model selection
- Persisted model artifact for repeatable inference
- Interactive prediction via Streamlit UI
- Programmatic prediction via Flask API
- Basic automated tests for training and inference validation

### Intended Audience

- ML beginners learning production-like project structure
- Developers integrating ML predictions into web/API products
- Reviewers evaluating practical machine learning engineering fundamentals

---

## Setup and Run Guide

This guide covers local setup, model training, and running both user interfaces (Streamlit + Flask API).

### 1. Prerequisites

- Windows, macOS, or Linux
- Python 3.11+
- `pip` available in your Python environment

### 2. Install Dependencies

From the project root (`ml_project/`), install requirements:

- `flask`
- `pandas`
- `pytest`
- `joblib`
- `scikit-learn`
- `streamlit`
- `matplotlib`
- `seaborn`

### 3. Dataset Expectations

The default training script expects:

- `data/house_prices (1).csv`

If the file is missing, training cannot start.

### 4. Train the Model

Run the training entrypoint (`main.py`) to:

1. Load dataset
2. Print quick EDA
3. Build preprocessing pipeline
4. Train candidate regressors
5. Select best model by RMSE
6. Save artifact to `models/model.pkl`

### 5. Run the Streamlit App

Launch `app.py` with Streamlit.

What to expect:

- Auto-generated form fields based on dataset columns
- Validation aligned to model feature schema
- Prediction output with formatted metric display

### 6. Run the Flask API

Start the Flask app from `src/web_app.py`.

Available routes:

- `GET /` – simple landing page
- `GET /health` – health check
- `POST /predict` – prediction endpoint

### 7. Quick Validation Flow

Recommended smoke test order:

1. Train model
2. Call `GET /health`
3. Send sample payload to `POST /predict`
4. Run Streamlit and submit one prediction

### 8. Common Setup Issues

#### Model file missing

Train first so `models/model.pkl` exists.

#### Serialization mismatch after package updates

Re-train model to regenerate a compatible artifact.

#### Invalid prediction payload

Ensure the request includes all required feature columns used during training.

---

## Architecture and Design

### Overview

This project follows a clean, modular architecture to separate concerns:

- data loading/preprocessing
- model training/selection
- inference logic
- presentation layers (Streamlit + Flask)
- automated tests

### Component Map

#### Data Layer
- `data/house_prices (1).csv`
- Loaded through `src/data_preprocessing.py`

#### ML Core
- `src/data_preprocessing.py`
  - Resolves target column
  - Splits features/target
  - Builds `ColumnTransformer` pipelines
- `src/model_training.py`
  - Trains and evaluates candidate models
  - Selects best model by RMSE
  - Saves model artifact with `joblib`
- `src/model_inference.py`
  - Loads model artifact
  - Normalizes/validates input schema
  - Returns single float prediction

#### Interface Layer
- `app.py` (Streamlit)
  - Dynamic form generation from dataset schema
  - Prediction UI and result rendering
- `src/web_app.py` (Flask)
  - REST-style endpoints for health + prediction

#### Entrypoint
- `main.py`
  - End-to-end training script for local runs

#### Quality Layer
- `tests/test_model.py`
  - Artifact generation test
  - Numeric prediction output test

### Data and Control Flow

1. CSV data loaded into DataFrame
2. Target column detected (`target`, `Target`, `price`, `Price`, or fallback to last column)
3. Numeric and categorical features transformed in dedicated pipelines
4. Candidate regressors trained and evaluated
5. Best pipeline persisted to `models/model.pkl`
6. Inference services load artifact and execute predictions on structured input

### Design Decisions

- **Single-source preprocessing** in training pipeline avoids train/serve skew.
- **Schema-aware inference** catches missing fields early with clear errors.
- **Model selection by RMSE** aligns with regression objective and robustness.
- **Dual interfaces** support both end-users (Streamlit) and service integration (Flask).

### Extensibility

- Add more regressors in `_build_models()`
- Add experiment tracking (e.g., MLflow)
- Introduce CI for linting/testing gates
- Add model versioning and drift monitoring

---

## Training and Inference Documentation

### Training Workflow

Training logic lives in `src/model_training.py` and is orchestrated by `main.py`.

#### Candidate Models

- Linear Regression
- Random Forest Regressor (`n_estimators=200`, `random_state=42`)

#### Preprocessing

Defined in `src/data_preprocessing.py`:

- Numeric features:
  - Mean imputation
  - Standard scaling
- Categorical features:
  - Most-frequent imputation
  - One-hot encoding (`handle_unknown="ignore"`)

#### Split and Evaluation

- Train/test split: `test_size=0.2`, `random_state=42`
- Metrics:
  - MAE
  - RMSE
  - $R^2$

Best model is selected by:

1. Lowest RMSE
2. Highest $R^2$ as tie-breaker

#### Artifact Output

Saved model pipeline:

- `models/model.pkl`

The artifact contains preprocessing + estimator in one pipeline.

### Inference Workflow

Inference logic in `src/model_inference.py`:

1. Load model artifact from disk
2. Infer expected feature columns from trained preprocessor
3. Validate incoming payload fields
4. Reindex payload to expected schema
5. Predict and return a `float`

### Error Handling Strategy

Inference explicitly raises clear exceptions for:

- missing model file
- wrong input type
- missing required fields

Flask and Streamlit layers catch these and return user-friendly messages.

### Operational Recommendation

After changing dependencies or schema:

1. Retrain model
2. Re-run tests
3. Validate both Streamlit and API prediction paths

---

## API Reference

Base application module: `src/web_app.py`

### `GET /`

Renders the default landing page.

#### Response

- Status: `200 OK`
- Content-Type: `text/html`

### `GET /health`

Health-check endpoint for uptime monitoring.

#### Response

- Status: `200 OK`
- Body:

```json
{
  "status": "ok"
}
```

### `POST /predict`

Generates a regression prediction from JSON feature input.

#### Request

- Content-Type: `application/json`
- Body must be a JSON object of feature names to values

#### Example Request Body

```json
{
  "Property_ID": "PROP0001",
  "Area": 3712,
  "Bedrooms": 4,
  "Bathrooms": 3,
  "Age": 36,
  "Location": "Rural",
  "Property_Type": "House"
}
```

#### Success Response

- Status: `200 OK`

```json
{
  "prediction": 23785000.42,
  "input": {
    "Property_ID": "PROP0001",
    "Area": 3712,
    "Bedrooms": 4,
    "Bathrooms": 3,
    "Age": 36,
    "Location": "Rural",
    "Property_Type": "House"
  }
}
```

#### Error Responses

- `400 Bad Request`
  - Invalid/missing JSON
  - Non-object payload
  - Missing required fields
  - File/model or validation issue surfaced from inference layer
- `500 Internal Server Error`
  - Unexpected prediction failure

### Notes for Integrators

- Keep payload keys aligned with training feature columns.
- Retrain model if package upgrades break artifact compatibility.
- Use `GET /health` for container/service readiness probes.

---

## Testing Guide

Test file: `tests/test_model.py`

### Scope Covered

Current automated tests verify:

1. Model artifact creation during training
2. Numeric prediction output from inference

### Test Cases

#### `test_train_model_creates_artifact`

- Trains model from dataset
- Saves to temporary path
- Asserts artifact exists and has `.pkl` extension

#### `test_predict_returns_numeric_value`

- Trains model to temporary path
- Sends sample inference payload
- Asserts prediction type is `float`

### Execution

Run the test suite with `pytest` from project root.

### Quality Recommendations

To strengthen coverage further:

- Add tests for missing payload fields
- Add tests for empty DataFrame and invalid target type
- Add API route tests using Flask test client
- Add regression tolerance checks for model quality thresholds

---

## Screenshots (SS) Guide

Store all project screenshots in `docs/screenshots/`.

These assets are referenced from the root `README.md`, so keep filenames exact.

### Required Screenshot Files

1. `streamlit_prediction_form.png`
   - Streamlit page with visible input form fields
2. `streamlit_prediction_result.png`
   - Streamlit page after a successful prediction
3. `api_response.png`
   - API client response for `POST /predict` (success case)

### Capture Standards

- Use PNG format.
- Keep resolution readable (recommended width: at least 1280px).
- Avoid cropping out key UI elements (title, form inputs, output).
- Do not include sensitive or personal data in payload screenshots.
- Use the same theme and zoom level across captures for consistency.

### Suggested Capture Workflow

1. Run training and confirm `models/model.pkl` exists.
2. Open Streamlit app and capture form view.
3. Submit sample values and capture prediction result.
4. Call Flask `POST /predict` via Postman/cURL/Thunder Client and capture response.
5. Save files with exact names listed above.

### Validation Checklist

- [ ] All three required files exist.
- [ ] Filenames exactly match documentation references.
- [ ] Text in screenshots is readable at normal zoom.
- [ ] Screenshots reflect the current project UI/API behavior.
