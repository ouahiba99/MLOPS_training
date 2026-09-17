"""
The single inference entrypoint. Both the CLI and the FastAPI routes call
this — logic lives in one place only.

get_model() loads from the MLflow Model Registry (Task 3, item 5) via the
"champion" alias, falling back to the local DVC-tracked artifact if the
registry is unreachable — see get_model()'s docstring.

Output field names match serving_config.json's contract: late_probability
is the primary output; predicted_late is a policy threshold applied on top
and can change without retraining (see config.model.decision_threshold).
The score is NOT a calibrated probability — see
config.model.score_is_calibrated_probability — so don't present it as one
downstream.

Also records the prediction-distribution metrics and the structured
prediction log (Task 3, item 10) — see src/metrics.py and
src/prediction_log.py for what's captured and why.
"""

import time
from typing import Any

import pandas as pd

from src.config import config
from src.data_access import load_joblib_artifact
from src.expectations import validate_data_quality
from src.features import build_features
from src.logger import get_logger
from src.metrics import (
    DATA_QUALITY_WARNING_COUNT,
    LATE_PROBABILITY,
    MODEL_SOURCE,
    PREDICTED_LATE_COUNT,
)
from src.model_registry import load_registered_model
from src.prediction_log import log_prediction
from src.validation import validate_order_payload

logger = get_logger(__name__)

_model = None  # loaded lazily, once per process


def get_model():
    """Loads from the MLflow registry first (Task 3, item 5) — falls back
    to the local DVC-tracked artifact only if the registry is unreachable
    or the model/alias isn't registered yet, so the service degrades
    gracefully instead of hard-failing on a registry outage."""
    global _model
    if _model is None:
        try:
            _model = load_registered_model()
            MODEL_SOURCE.set(1)
            logger.info(
                "loaded model from MLflow registry: %s@%s",
                config.mlflow.registered_model_name,
                config.mlflow.model_alias,
            )
        except Exception as e:
            logger.warning(
                "could not load model from MLflow registry (%s); "
                "falling back to local artifact",
                e,
            )
            _model = load_joblib_artifact("model")
            MODEL_SOURCE.set(0)
    return _model


def predict_order(payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()

    validate_order_payload(payload)
    raw_df = pd.DataFrame([payload])
    data_quality_warnings = validate_data_quality(raw_df)
    features = build_features(raw_df)

    model = get_model()
    late_probability = float(model.predict_proba(features)[:, 1][0])
    predicted_late = late_probability >= config.model.decision_threshold

    result = {
        "late_probability": late_probability,
        "predicted_late": predicted_late,
        "model_version": config.model.version,
        "data_quality_warnings": data_quality_warnings,
    }

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "prediction request | input=%s | output=%s | latency_ms=%.2f | "
        "model_version=%s",
        payload,
        result,
        round(latency_ms, 2),
        config.model.version,
    )
    if data_quality_warnings:
        logger.warning("data quality warnings for request: %s", data_quality_warnings)

    LATE_PROBABILITY.observe(late_probability)
    PREDICTED_LATE_COUNT.labels(predicted_late=str(predicted_late)).inc()
    for warning in data_quality_warnings:
        column = warning.split(":", 1)[0]
        DATA_QUALITY_WARNING_COUNT.labels(column=column).inc()

    log_prediction(payload, result)

    return result
