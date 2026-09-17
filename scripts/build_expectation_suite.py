"""
Builds the Great Expectations suite for incoming order payloads and saves
it as JSON to great_expectations/expectations/order_payload_suite.json.

Run this to regenerate the suite after changing thresholds below (e.g.
once you have real percentiles from your training data to replace the
placeholder sanity bounds marked TODO):

    python scripts/build_expectation_suite.py

Two severities are used deliberately, matching Task 3 item 4's "decide
what the service does when validation fails" — reject vs flag:

- severity="critical": structurally invalid data. src/expectations.py
  raises ValidationError (-> API 400) if any of these fail.
- severity="warning": statistically unusual but not invalid — e.g. a
  real Brazilian state that was rare/absent in training. Logged and
  returned in the response as data_quality_warnings; the request still
  gets a prediction.

No expectation uses a "default" (substituted) value: every field in this
schema is either required-and-validated or a soft warning — there's no
column where silently substituting a value is safer than rejecting or
flagging, so "default" isn't used here. Documented, not just omitted.
"""

import json
from pathlib import Path

import great_expectations as gx

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "great_expectations"
    / "expectations"
    / "order_payload_suite.json"
)

# All 27 Brazilian state codes — a real state Notebook 4's target encoder /
# one-hot encoder rarely or never saw is still a REAL state (flag, don't
# reject). Anything outside this list is malformed (reject).
ALL_BR_STATES = [
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
]

# The states the fitted OneHotEncoder actually gave their own column to
# (from feature_names.json) — everything else valid falls into sklearn's
# "infrequent" bucket, meaning the model saw few/no such examples.
WELL_REPRESENTED_CUSTOMER_STATES = [
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MG",
    "MS",
    "MT",
    "PA",
    "PB",
    "PE",
    "PI",
    "PR",
    "RJ",
    "RN",
    "RS",
    "SC",
    "SP",
]
WELL_REPRESENTED_SELLER_STATES = ["BA", "DF", "MG", "PR", "RJ", "RS", "SC", "SP"]

NUMERIC_MIN_1_COLS = [
    "item_count",
    "unique_products",
    "unique_sellers",
    "payment_count",
    "max_payment_installments",
]
NUMERIC_MIN_0_COLS = [
    "total_item_value",
    "total_freight_value",
    "total_payment_value",
    "distance_km",
]

# Placeholder soft upper bounds (WARNING only) — rough sanity caps, not
# derived from your actual training-data percentiles. Replace these once
# you've pulled real p99s from Notebook 3/4's data and rerun this script.
SOFT_UPPER_BOUNDS = {
    "item_count": 20,
    "total_item_value": 2000.0,
    "total_freight_value": 200.0,
    "total_payment_value": 3000.0,
    "max_payment_installments": 24,
    "distance_km": 4000.0,  # Brazil's north-south span is roughly this
}


def build_suite() -> "gx.ExpectationSuite":
    suite = gx.ExpectationSuite(name="order_payload_suite")
    E = gx.expectations

    # --- Missing rate: every required field must be present (CRITICAL) ---
    for col in [
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
    ]:
        suite.add_expectation(
            E.ExpectColumnValuesToNotBeNull(column=col, severity="critical")
        )

    # --- Types / parseability (CRITICAL) ---
    # Bad dates would otherwise silently become NaT under
    # pd.to_datetime(errors="coerce") in engineer_features() and get quietly
    # imputed downstream — catch it here instead.
    for col in ["order_purchase_timestamp", "order_estimated_delivery_date"]:
        suite.add_expectation(
            E.ExpectColumnValuesToBeDateutilParseable(column=col, severity="critical")
        )

    suite.add_expectation(
        E.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="order_estimated_delivery_date",
            column_B="order_purchase_timestamp",
            severity="critical",
        )
    )

    # --- Ranges (CRITICAL: structural validity) ---
    for col in NUMERIC_MIN_1_COLS:
        suite.add_expectation(
            E.ExpectColumnValuesToBeBetween(
                column=col, min_value=1, severity="critical"
            )
        )
    for col in NUMERIC_MIN_0_COLS:
        suite.add_expectation(
            E.ExpectColumnValuesToBeBetween(
                column=col, min_value=0, severity="critical"
            )
        )

    # Logical consistency: can't have more distinct products/sellers than items.
    suite.add_expectation(
        E.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="item_count",
            column_B="unique_products",
            or_equal=True,
            severity="critical",
        )
    )
    suite.add_expectation(
        E.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="item_count",
            column_B="unique_sellers",
            or_equal=True,
            severity="critical",
        )
    )

    # --- Ranges (WARNING: soft sanity caps — see SOFT_UPPER_BOUNDS docstring) ---
    for col, upper in SOFT_UPPER_BOUNDS.items():
        suite.add_expectation(
            E.ExpectColumnValuesToBeBetween(
                column=col, max_value=upper, severity="warning"
            )
        )

    # --- Allowed categories ---
    # CRITICAL: must be a real Brazilian state at all.
    suite.add_expectation(
        E.ExpectColumnValuesToBeInSet(
            column="customer_state", value_set=ALL_BR_STATES, severity="critical"
        )
    )
    suite.add_expectation(
        E.ExpectColumnValuesToBeInSet(
            column="seller_state", value_set=ALL_BR_STATES, severity="critical"
        )
    )
    # WARNING: valid state, but rare/absent in what the model was trained on.
    suite.add_expectation(
        E.ExpectColumnValuesToBeInSet(
            column="customer_state",
            value_set=WELL_REPRESENTED_CUSTOMER_STATES,
            severity="warning",
        )
    )
    suite.add_expectation(
        E.ExpectColumnValuesToBeInSet(
            column="seller_state",
            value_set=WELL_REPRESENTED_SELLER_STATES,
            severity="warning",
        )
    )

    return suite


def main():
    # An active context is required before ExpectationSuite.add_expectation() works.
    gx.get_context(mode="ephemeral")
    suite = build_suite()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(suite.to_json_dict(), f, indent=2)
    print(f"Wrote {len(suite.expectations)} expectations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
