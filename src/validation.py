"""
Input validation before a request reaches the model. Two checks:

1. Reject any of the "forbidden" fields — post-outcome / leakage columns
   (delivery dates, reviews, approval time) that must never influence a
   checkout-time prediction. Loaded from feature_config.json when present;
   falls back to the literal list from serving_config.json otherwise.
2. Basic completeness check on the raw columns engineer_features() and the
   target encoder actually need.

Task 3 item 4 replaces/extends this with Great Expectations for real
column types, ranges, and allowed-category checks.
"""

from typing import Any

from src.data_access import load_feature_config

# Mirrors feature_config.json's api_forbidden_fields (confirmed via
# serving_config.json's note) — used only if feature_config.json isn't in
# models/artifacts/ yet.
DEFAULT_FORBIDDEN_FIELDS = [
    "is_late",
    "delivery_delay_days",
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "review_count",
    "average_review_score",
    "order_approved_at",
    "approval_delay_hours",
]

# Raw columns needed to run engineer_features() + target encoding + OHE.
# Confirmed via Notebook 4's explicit prediction-time leakage audit
# (KNOWN_SAFE) — these are the only columns Notebook 4 classifies as safe
# at the checkout-time prediction point, and cross-checked against
# Notebook 5's LEAKAGE_COLS drop list. The accounting is exact: target (1)
# + safe (17) + post-outcome (5) + ID/ambiguous (5) = 28 raw columns total,
# with nothing left unclassified.
REQUIRED_FIELDS = [
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "customer_state",
    "customer_city",
    "customer_zip_code_prefix",
    "seller_state",
    "seller_city",
    "seller_zip_code_prefix",
    "item_count",
    "unique_products",
    "unique_sellers",
    "total_item_value",
    "total_freight_value",
    "payment_count",
    "total_payment_value",
    "max_payment_installments",
    "distance_km",
]

# Not required, not rejected — order_status, order_id, customer_id, and
# customer_unique_id are accepted if sent (e.g. as metadata) but are
# silently dropped in preprocessing.py via feature_config's
# leakage_cols_dropped; they were excluded from the model's features by
# Notebook 5 per Notebook 4's audit.


class ValidationError(Exception):
    """Raised on bad input so the API can return a clean 4xx, not a crash."""


def _forbidden_fields() -> list[str]:
    try:
        return load_feature_config()["api_forbidden_fields"]
    except FileNotFoundError:
        return DEFAULT_FORBIDDEN_FIELDS


def validate_order_payload(payload: dict[str, Any]) -> None:
    if not payload:
        raise ValidationError("Empty payload")

    forbidden_present = [f for f in _forbidden_fields() if f in payload]
    if forbidden_present:
        raise ValidationError(
            f"These fields are not available at prediction time and must "
            f"not be sent: {forbidden_present}"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")
