from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "notebook06-histgbm-checkout-v2"
    assert body["score_is_calibrated_probability"] is False


def test_predict_rejects_incomplete_payload():
    # Missing required fields -> 400, not a crash.
    response = client.post("/predict", json={"order_id": "abc123"})
    assert response.status_code == 400


def test_predict_returns_a_real_prediction():
    # With preprocessor.joblib / target_encoder.joblib / feature_config.json
    # in place, this should return a real prediction, not a 503.
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
    body = response.json()
    assert "late_probability" in body
    assert 0.0 <= body["late_probability"] <= 1.0
    assert "predicted_late" in body
    assert body["model_version"] == "notebook06-histgbm-checkout-v2"
    assert body["data_quality_warnings"] == []  # SP/RJ are both well-represented states


def test_predict_flags_rare_state_but_still_predicts():
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
        "customer_city": "rio branco",
        "customer_state": "AC",  # real state, rare in training -> flag, not reject
        "seller_zip_code_prefix": "20000",
        "seller_city": "rio de janeiro",
        "seller_state": "RJ",
        "distance_km": 430.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["data_quality_warnings"]) >= 1


def test_predict_rejects_impossible_state():
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
        "customer_city": "nowhere",
        "customer_state": "ZZ",  # not a real state -> reject
        "seller_zip_code_prefix": "20000",
        "seller_city": "rio de janeiro",
        "seller_state": "RJ",
        "distance_km": 430.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
