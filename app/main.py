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
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import config
from src.logger import get_logger
from src.metrics import REQUEST_COUNT, REQUEST_LATENCY
from src.model_registry import get_champion_version_info
from src.predict import predict_order
from src.validation import ValidationError

logger = get_logger(__name__)

app = FastAPI(title="Olist Late Delivery Prediction Service")


@app.middleware("http")
async def track_request_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(
        time.perf_counter() - start
    )
    REQUEST_COUNT.labels(
        endpoint=request.url.path, status_code=response.status_code
    ).inc()
    return response


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}


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
    return [predict(p) for p in payloads]
