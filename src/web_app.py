"""Flask web API for serving model predictions."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

from src.model_inference import predict

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render a simple landing page."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health() -> tuple[Any, int]:
    """Health-check endpoint for uptime monitoring."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict_route() -> tuple[Any, int]:
    """Predict from JSON input and return a structured response."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "JSON payload must be an object of feature names to values."}), 400

    try:
        prediction = predict(payload)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Unexpected server error during prediction."}), 500

    return jsonify({"prediction": float(prediction), "input": payload}), 200
