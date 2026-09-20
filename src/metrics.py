"""
Prometheus metrics (Task 3, item 10).

Two distinct kinds of metric here, and it matters which is which:

- Request-level metrics (REQUEST_COUNT, REQUEST_LATENCY) are standard API
  observability — request count, latency, error rate — wired in once via
  middleware (app/main.py) so every route is covered automatically,
  including any added later.

- Prediction-distribution metrics (LATE_PROBABILITY, PREDICTED_LATE_COUNT,
  DATA_QUALITY_WARNING_COUNT) are model observability, not API
  observability: they're what "watch for drift" actually means here.
  is_late's base rate in training was ~9% (target_encoder_global_mean in
  feature_config.json) — if PREDICTED_LATE_COUNT's ratio drifts far from
  that over time with no change to the service, that's a signal the input
  distribution has shifted, not that deliveries actually got worse.
  DATA_QUALITY_WARNING_COUNT closes the loop with item 4: a rising rate of
  "rare state" warnings, for instance, is an early, cheap drift signal —
  it's flagging the same kind of shift LATE_PROBABILITY would eventually
  show, but sooner and more directly tied to *which* input changed.
"""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

LATE_PROBABILITY = Histogram(
    "prediction_late_probability",
    "Distribution of late_probability scores returned by /predict",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

PREDICTED_LATE_COUNT = Counter(
    "prediction_predicted_late_total",
    "Count of predictions by the thresholded predicted_late outcome",
    ["predicted_late"],
)

DATA_QUALITY_WARNING_COUNT = Counter(
    "data_quality_warning_total",
    "Count of Great Expectations warning-severity failures by column",
    ["column"],
)

MODEL_SOURCE = Gauge(
    "model_using_registry",
    "1 if the currently-loaded model came from the MLflow registry, "
    "0 if it fell back to the local DVC-tracked artifact. A Gauge, not a "
    "Counter — get_model() only runs once per process (the result is "
    "cached), so this is persistent state to alert on, not a rate of "
    "events that could roll out of an alerting window.",
)

# Pipeline-level breakdown: measures time in validation, GE,
# features, model, and logging
PIPELINE_STAGE_LATENCY = Histogram(
    "prediction_pipeline_stage_latency_seconds",
    "Latency breakdown by prediction pipeline stage in seconds",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Concurrency & traffic telemetry
IN_FLIGHT_REQUESTS = Gauge(
    "api_requests_in_flight",
    "Number of concurrent HTTP requests currently in flight",
    ["endpoint"],
)

BATCH_SIZE = Histogram(
    "prediction_batch_size",
    "Distribution of batch sizes in /predict/batch requests",
    buckets=(1, 5, 10, 25, 50, 100, 250),
)

# Model configuration & governance metadata
MODEL_DECISION_THRESHOLD = Gauge(
    "model_decision_threshold",
    "Active F1-optimal decision threshold for classification",
)

MODEL_INFO = Gauge(
    "model_info",
    "Metadata for the active model and serving configuration",
    ["model_name", "model_version", "model_alias"],
)

# Real-time input feature distribution tracking for drift detection
INPUT_DISTANCE_KM = Histogram(
    "prediction_input_distance_km",
    "Distribution of customer-seller distance in km from incoming requests",
    buckets=(50, 100, 250, 500, 750, 1000, 1500, 2000, 3000),
)

INPUT_TOTAL_PAYMENT = Histogram(
    "prediction_input_total_payment_value",
    "Distribution of total payment value from incoming requests",
    buckets=(20, 50, 100, 200, 500, 1000, 2000, 5000),
)

INPUT_FREIGHT_VALUE = Histogram(
    "prediction_input_freight_value",
    "Distribution of freight value from incoming requests",
    buckets=(10, 20, 30, 50, 75, 100, 150, 200),
)
