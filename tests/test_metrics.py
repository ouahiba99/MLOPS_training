"""
Task 3, item 10 test coverage: metrics actually change when they should.

Uses REGISTRY.get_sample_value(), the standard prometheus_client way to
read a metric's current value in tests. Reads a before/after delta rather
than an absolute value, since these are process-global counters/histograms
shared across the whole test session — other tests increment them too.
"""

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.main import app

client = TestClient(app)


def test_health_request_increments_request_count():
    before = (
        REGISTRY.get_sample_value(
            "api_requests_total", {"endpoint": "/health", "status_code": "200"}
        )
        or 0
    )
    client.get("/health")
    after = REGISTRY.get_sample_value(
        "api_requests_total", {"endpoint": "/health", "status_code": "200"}
    )
    assert after == before + 1


def test_predict_records_late_probability_and_predicted_late():
    payload = {
        "order_purchase_timestamp": "2026-01-01T10:00:00",
        "order_estimated_delivery_date": "2026-01-15T00:00:00",
        "total_freight_value": 20.0,
        "total_item_value": 100.0,
        "total_payment_value": 120.0,
        "payment_count": 1,
        "max_payment_installments": 1,
        "item_count": 1,
        "unique_sellers": 1,
        "unique_products": 1,
        "customer_zip_code_prefix": "01001",
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "seller_zip_code_prefix": "20000",
        "seller_city": "rio de janeiro",
        "seller_state": "RJ",
        "distance_km": 430.0,
    }
    count_before = REGISTRY.get_sample_value("prediction_late_probability_count") or 0

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    predicted_late = str(response.json()["predicted_late"])

    late_before = (
        REGISTRY.get_sample_value(
            "prediction_predicted_late_total", {"predicted_late": predicted_late}
        )
        or 0
    )
    response2 = client.post("/predict", json=payload)
    assert response2.status_code == 200

    count_after = REGISTRY.get_sample_value("prediction_late_probability_count")
    late_after = REGISTRY.get_sample_value(
        "prediction_predicted_late_total", {"predicted_late": predicted_late}
    )
    assert count_after is not None
    assert late_after is not None
    assert count_after >= count_before + 1
    assert late_after == late_before + 1


def test_model_using_registry_gauge_is_set_after_a_prediction():
    client.post(
        "/predict",
        json={
            "order_purchase_timestamp": "2026-01-01T10:00:00",
            "order_estimated_delivery_date": "2026-01-15T00:00:00",
            "total_freight_value": 20.0,
            "total_item_value": 100.0,
            "total_payment_value": 120.0,
            "payment_count": 1,
            "max_payment_installments": 1,
            "item_count": 1,
            "unique_sellers": 1,
            "unique_products": 1,
            "customer_zip_code_prefix": "01001",
            "customer_city": "sao paulo",
            "customer_state": "SP",
            "seller_zip_code_prefix": "20000",
            "seller_city": "rio de janeiro",
            "seller_state": "RJ",
            "distance_km": 430.0,
        },
    )
    # A Gauge, not a Counter — see src/metrics.py's docstring for why that
    # matters for alerting. Value is 0 or 1 depending on which path
    # get_model() actually resolved this test run.
    value = REGISTRY.get_sample_value("model_using_registry")
    assert value in (0.0, 1.0)


def test_metrics_endpoint_returns_prometheus_text_format():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
    assert "prediction_pipeline_stage_latency_seconds" in response.text
    assert "prediction_input_distance_km" in response.text


def test_pipeline_stage_latency_recorded():
    payload = {
        "order_purchase_timestamp": "2026-01-01T10:00:00",
        "order_estimated_delivery_date": "2026-01-15T00:00:00",
        "total_freight_value": 20.0,
        "total_item_value": 100.0,
        "total_payment_value": 120.0,
        "payment_count": 1,
        "max_payment_installments": 1,
        "item_count": 1,
        "unique_sellers": 1,
        "unique_products": 1,
        "customer_zip_code_prefix": "01001",
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "seller_zip_code_prefix": "20000",
        "seller_city": "rio de janeiro",
        "seller_state": "RJ",
        "distance_km": 430.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    for stage in ["validation", "expectations", "features", "inference", "logging"]:
        val = REGISTRY.get_sample_value(
            "prediction_pipeline_stage_latency_seconds_count", {"stage": stage}
        )
        assert val is not None and val >= 1.0


def test_input_feature_histograms_recorded():
    payload = {
        "order_purchase_timestamp": "2026-01-01T10:00:00",
        "order_estimated_delivery_date": "2026-01-15T00:00:00",
        "total_freight_value": 35.5,
        "total_item_value": 150.0,
        "total_payment_value": 185.5,
        "payment_count": 1,
        "max_payment_installments": 2,
        "item_count": 1,
        "unique_sellers": 1,
        "unique_products": 1,
        "customer_zip_code_prefix": "01001",
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "seller_zip_code_prefix": "20000",
        "seller_city": "rio de janeiro",
        "seller_state": "RJ",
        "distance_km": 520.0,
    }
    dist_before = REGISTRY.get_sample_value("prediction_input_distance_km_count") or 0.0
    client.post("/predict", json=payload)
    dist_after = REGISTRY.get_sample_value("prediction_input_distance_km_count")
    assert dist_after is not None
    assert dist_after >= dist_before + 1.0


def test_batch_size_metric_recorded():
    payload = {
        "order_purchase_timestamp": "2026-01-01T10:00:00",
        "order_estimated_delivery_date": "2026-01-15T00:00:00",
        "total_freight_value": 20.0,
        "total_item_value": 100.0,
        "total_payment_value": 120.0,
        "payment_count": 1,
        "max_payment_installments": 1,
        "item_count": 1,
        "unique_sellers": 1,
        "unique_products": 1,
        "customer_zip_code_prefix": "01001",
        "customer_city": "sao paulo",
        "customer_state": "SP",
        "seller_zip_code_prefix": "20000",
        "seller_city": "rio de janeiro",
        "seller_state": "RJ",
        "distance_km": 430.0,
    }
    batch_before = REGISTRY.get_sample_value("prediction_batch_size_count") or 0.0
    response = client.post("/predict/batch", json=[payload, payload])
    assert response.status_code == 200
    batch_after = REGISTRY.get_sample_value("prediction_batch_size_count")
    assert batch_after is not None
    assert batch_after >= batch_before + 1.0


def test_health_probes():
    live_res = client.get("/health/live")
    assert live_res.status_code == 200
    assert live_res.json()["status"] == "alive"

    ready_res = client.get("/health/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["status"] == "ready"
    assert ready_res.json()["model_loaded"] is True


def test_monitoring_summary_endpoint():
    response = client.get("/monitoring/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "model" in data
    assert "inference_stats" in data
    assert data["model"]["version"] is not None
