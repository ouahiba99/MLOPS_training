"""
Runs incoming order data through the Great Expectations suite defined in
great_expectations/expectations/order_payload_suite.json (built by
scripts/build_expectation_suite.py).

This is the second, fuller validation layer. src/validation.py's forbidden-
field and required-field checks run first (cheap, fast-fail on obviously
wrong requests); this suite checks types, ranges, logical consistency, and
allowed categories on whatever's left, using Notebook 4's confirmed raw
schema.

Task 3 item 4 asks: "Decide what the service does when validation fails —
reject, flag, or default." The answer here is a deliberate split by
severity, not one policy for everything:

- severity="critical": structurally invalid data (a state that doesn't
  exist, a negative distance, an estimated delivery date before the
  purchase date). validate_data_quality() raises ValidationError -> the
  API returns a 400. Rejected, not predicted on.
- severity="warning": statistically unusual but not invalid — e.g. a real
  Brazilian state the model rarely or never saw in training. Logged and
  surfaced in the response as data_quality_warnings; the request still
  gets a prediction, because rejecting a valid state the model can still
  score (via its infrequent-category bucket) would be overly strict.

No expectation uses "default" (silently substituting a value) — every
field here is either required-and-validated or a soft warning. There's no
column where inventing a value is safer than rejecting or flagging it, so
that policy option is deliberately unused, not just forgotten.

The suite JSON is loaded once at import time (build_suite_from_json /
get_validator); only get_batch()/validate() run per request.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import great_expectations as gx
from src.validation import ValidationError

SUITE_PATH = (
    Path(__file__).resolve().parents[1]
    / "great_expectations"
    / "expectations"
    / "order_payload_suite.json"
)


def load_suite() -> gx.ExpectationSuite:
    with open(SUITE_PATH) as f:
        suite_dict = json.load(f)
    return gx.ExpectationSuite(
        name=suite_dict["name"], expectations=suite_dict["expectations"]
    )


def _describe(result) -> str:
    """One readable line for a failed expectation result, e.g.
    'customer_state: expect_column_values_to_be_in_set failed (got: ZZ)'."""
    kwargs = result.expectation_config.kwargs
    column = (
        kwargs.get("column") or f"{kwargs.get('column_A')}/{kwargs.get('column_B')}"
    )
    exp_type = result.expectation_config.type
    unexpected = result.result.get("partial_unexpected_list")
    detail = f" (got: {unexpected})" if unexpected else ""
    return f"{column}: {exp_type} failed{detail}"


@dataclass
class DataQualityOutcome:
    success: bool  # True unless a CRITICAL expectation failed
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class GreatExpectationsValidator:
    """Builds the GX context/datasource/suite once; validates one row at a time."""

    def __init__(self):
        self.context = gx.get_context(mode="ephemeral")
        # Silences the per-call "Calculating Metrics" progress bar, which
        # would otherwise spam a server's stdout on every request.
        self.context.variables.progress_bars = {
            "globally": False,
            "metric_calculations": False,
        }
        datasource = self.context.data_sources.add_pandas("orders")
        asset = datasource.add_dataframe_asset(name="orders_asset")
        self.batch_definition = asset.add_batch_definition_whole_dataframe(
            "orders_batch"
        )
        self.suite = self.context.suites.add(load_suite())

    def check(self, df: pd.DataFrame) -> DataQualityOutcome:
        batch = self.batch_definition.get_batch(batch_parameters={"dataframe": df})
        result = batch.validate(self.suite)

        critical_failures, warnings = [], []
        for r in result.results:
            if r.success:
                continue
            message = _describe(r)
            if (
                r.expectation_config.severity is not None
                and r.expectation_config.severity.value == "warning"
            ):
                warnings.append(message)
            else:
                critical_failures.append(message)

        return DataQualityOutcome(
            success=len(critical_failures) == 0,
            critical_failures=critical_failures,
            warnings=warnings,
        )


_validator: GreatExpectationsValidator | None = None  # built lazily, once per process


def get_validator() -> GreatExpectationsValidator:
    global _validator
    if _validator is None:
        _validator = GreatExpectationsValidator()
    return _validator


def check_data_quality(df: pd.DataFrame) -> DataQualityOutcome:
    return get_validator().check(df)


def validate_data_quality(df: pd.DataFrame) -> list[str]:
    """Raises ValidationError on any critical failure. Returns the list of
    non-blocking warnings (empty if none) — the caller (src/predict.py)
    logs and surfaces these in the response rather than rejecting."""
    outcome = check_data_quality(df)
    if not outcome.success:
        raise ValidationError(f"Data quality check failed: {outcome.critical_failures}")
    return outcome.warnings
