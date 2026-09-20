"""
FastAPI service for the Olist late-delivery model.

Health and model-info routes work out of the box. /predict is wired end
to end: forbidden-field/required-field checks (src/validation.py), then a
Great Expectations data-quality pass (src/expectations.py) with a
critical-vs-warning severity split — critical failures reject the
request (400), warnings ride along in the response's
data_quality_warnings field and the prediction still happens. The model
itself loads from the MLflow Model Registry (src/model_registry.py,
Task 3 item 5), falling back to the local DVC-tracked artifact if the
registry is unreachable.

/metrics (Task 3, item 10) exposes request count/latency/error rate via
a middleware that wraps every route automatically — see src/metrics.py
for what's tracked and why prediction-distribution metrics are a
separate concern from these request-level ones.
"""

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from src.config import config
from src.logger import get_logger
from src.metrics import (
    BATCH_SIZE,
    IN_FLIGHT_REQUESTS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.model_registry import get_champion_version_info
from src.predict import get_model, predict_order
from src.validation import ValidationError

logger = get_logger(__name__)

app = FastAPI(title="Olist Late Delivery Prediction Service")

_START_TIME = time.time()


@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    endpoint = request.url.path
    IN_FLIGHT_REQUESTS.labels(endpoint=endpoint).inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        REQUEST_COUNT.labels(endpoint=endpoint, status_code=response.status_code).inc()
        return response
    finally:
        IN_FLIGHT_REQUESTS.labels(endpoint=endpoint).dec()


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/live")
def health_live():
    """Liveness probe: verifies the ASGI worker process is responsive."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/ready")
def health_ready():
    """Readiness probe: verifies the model artifact is loaded and operational."""
    try:
        model = get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {
            "status": "ready",
            "model_loaded": True,
            "model_version": config.model.version,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Readiness check failed: {e}")


@app.get("/monitoring/summary")
def monitoring_summary():
    """Human and dashboard-friendly JSON telemetry summary."""
    uptime_seconds = round(time.time() - _START_TIME, 1)
    model_source_val = REGISTRY.get_sample_value("model_using_registry")
    if model_source_val == 1.0:
        source_desc = "mlflow_registry"
    elif model_source_val == 0.0:
        source_desc = "local_fallback"
    else:
        source_desc = "uninitialized"

    late_total = (
        REGISTRY.get_sample_value(
            "prediction_predicted_late_total", {"predicted_late": "True"}
        )
        or 0.0
    )
    not_late_total = (
        REGISTRY.get_sample_value(
            "prediction_predicted_late_total", {"predicted_late": "False"}
        )
        or 0.0
    )
    total_predictions = late_total + not_late_total
    late_rate = (
        round(late_total / total_predictions, 4) if total_predictions > 0 else None
    )

    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "model": {
            "name": config.model.name,
            "version": config.model.version,
            "decision_threshold": config.model.decision_threshold,
            "source": source_desc,
        },
        "inference_stats": {
            "total_predictions": int(total_predictions),
            "predicted_late_count": int(late_total),
            "predicted_on_time_count": int(not_late_total),
            "predicted_late_rate": late_rate,
        },
    }


@app.get("/model/info")
def model_info():
    info = {
        "name": config.model.name,
        "version": config.model.version,
        "decision_threshold": config.model.decision_threshold,
        "score_is_calibrated_probability": config.model.score_is_calibrated_probability,
    }
    try:
        info["registry"] = get_champion_version_info()
    except Exception as e:
        info["registry"] = None
        info["registry_error"] = str(e)
    return info


@app.post("/predict")
def predict(payload: dict):
    try:
        return predict_order(payload)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal error")


@app.post("/predict/batch")
def predict_batch(payloads: list[dict]):
    BATCH_SIZE.observe(len(payloads))
    return [predict(p) for p in payloads]
