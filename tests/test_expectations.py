"""
Task 3, item 4 test coverage: column ranges, allowed categories, and the
reject-vs-flag policy split by severity.
"""

import pandas as pd
import pytest

from src.expectations import check_data_quality
from src.validation import ValidationError

BASE_ROW = {
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


def _row(**overrides):
    return pd.DataFrame([{**BASE_ROW, **overrides}])


def test_well_formed_payload_has_no_failures_or_warnings():
    outcome = check_data_quality(_row())
    assert outcome.success
    assert outcome.warnings == []


def test_rare_but_real_state_is_flagged_not_rejected():
    # AC is a real Brazilian state, just rare/absent in what the model
    # trained on (not one of the OHE's explicit columns) -> warning only.
    outcome = check_data_quality(_row(customer_state="AC"))
    assert outcome.success  # not rejected
    assert any("customer_state" in w for w in outcome.warnings)


def test_invalid_state_is_rejected():
    outcome = check_data_quality(_row(customer_state="ZZ"))
    assert not outcome.success
    assert any("customer_state" in f for f in outcome.critical_failures)


def test_negative_distance_is_rejected():
    outcome = check_data_quality(_row(distance_km=-5.0))
    assert not outcome.success


def test_zero_item_count_is_rejected():
    outcome = check_data_quality(_row(item_count=0))
    assert not outcome.success


def test_more_unique_products_than_items_is_rejected():
    # Logically impossible: can't have 3 distinct products across 1 item.
    outcome = check_data_quality(_row(item_count=1, unique_products=3))
    assert not outcome.success


def test_estimated_delivery_before_purchase_is_rejected():
    outcome = check_data_quality(
        _row(
            order_purchase_timestamp="2026-01-10T00:00:00",
            order_estimated_delivery_date="2026-01-05T00:00:00",
        )
    )
    assert not outcome.success


def test_unusually_high_value_is_flagged_not_rejected():
    outcome = check_data_quality(_row(total_item_value=5000.0))
    assert outcome.success
    assert any("total_item_value" in w for w in outcome.warnings)


def test_validate_data_quality_raises_on_critical_failure():
    from src.expectations import validate_data_quality

    with pytest.raises(ValidationError):
        validate_data_quality(_row(customer_state="ZZ"))
