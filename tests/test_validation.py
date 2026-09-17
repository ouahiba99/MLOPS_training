"""
Data/leakage tests (Task 3, item 6): schema completeness and the
forbidden-field rejection that keeps post-outcome columns out of a
checkout-time prediction. Runs standalone — falls back to
DEFAULT_FORBIDDEN_FIELDS when feature_config.json isn't in
models/artifacts/ yet.
"""

import pytest

from src.validation import REQUIRED_FIELDS, ValidationError, validate_order_payload

COMPLETE_PAYLOAD = {
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


def test_empty_payload_rejected():
    with pytest.raises(ValidationError):
        validate_order_payload({})


def test_complete_payload_passes():
    validate_order_payload(COMPLETE_PAYLOAD)  # should not raise


def test_missing_required_field_rejected():
    incomplete = {k: v for k, v in COMPLETE_PAYLOAD.items() if k != "distance_km"}
    with pytest.raises(ValidationError, match="distance_km"):
        validate_order_payload(incomplete)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "is_late",
        "delivery_delay_days",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "review_count",
        "average_review_score",
        "order_approved_at",
    ],
)
def test_post_outcome_fields_rejected(forbidden_field):
    payload = {**COMPLETE_PAYLOAD, forbidden_field: "anything"}
    with pytest.raises(ValidationError, match=forbidden_field):
        validate_order_payload(payload)


def test_required_fields_cover_notebook4_safe_columns():
    # Guards against silently drifting from the Notebook 4 audit.
    assert len(REQUIRED_FIELDS) == 17
