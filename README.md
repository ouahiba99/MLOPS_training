# Olist Late-Delivery — Inference Service

Turns Notebook 5 (feature engineering) and Notebook 6 (train/tune/select)
from the Qafza MLOps Training Task 2 into a production inference service:
same fitted objects, same predictions, now callable from an API.

## Structure

```
app/            FastAPI app (routes only — no business logic here)
config/         config.yaml — every path and parameter lives here
data/           local data (git-ignored; DVC-tracked once you export a split)
great_expectations/expectations/order_payload_suite.json  the GE suite (generated — see item 4 below)
models/
  artifacts/    model.joblib, preprocessor.joblib, etc. — DVC-tracked as one bundle
  artifacts.dvc DVC pointer file (this IS committed to git)
  results_summary.json, serving_config.json  Notebook 6's own output — read by scripts/log_to_mlflow.py
notebooks/      the 6 training notebooks from Task 2 — copy them in here
Dockerfile               API image (item 8)
docker/mlflow.Dockerfile MLflow tracking server image (item 8)
docker-compose.yml       postgres + mlflow + mlflow-init + api, one command (item 8)
.env.example             required env vars, no real secrets (copy to .env)
.github/workflows/ci.yml lint, format, dvc pull, register, test, build+push (item 9)
.pre-commit-config.yaml  fast local checks before every commit (item 9)
pyproject.toml           pins black/ruff config so CI matches local runs
monitoring/prometheus.yml  scrape config (item 10)
monitoring/alerts.yml      alerting rules, validated with promtool (item 10)
scripts/build_expectation_suite.py  regenerates the GE suite JSON
scripts/log_to_mlflow.py            logs a training run + registers the model (item 5)
scripts/evaluate_predictions.py     joins prediction logs against ground truth (item 10)
src/            the pipeline: config, logging, data access, validation,
                expectations (GE), preprocessing, feature building,
                model_registry (MLflow), predict, metrics + prediction_log (item 10)
tests/          pytest — unit + model + validation + expectations + registry + integration tests
requirements/   requirements.txt (runtime) and requirements-dev.txt (dev/test/dvc tools)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements/requirements-dev.txt
```

`models/artifacts/` now has everything needed for `/predict` to return real
predictions: `final_model.joblib`, `preprocessor.joblib`,
`target_encoder.joblib`, `feature_names.json`, `feature_config.json` — all
DVC-tracked as one bundle (see item 4 below). If you're cloning this fresh
and the files aren't there, run `dvc pull` instead of copying them by hand.

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the interactive API docs.

Routes: `GET /health`, `GET /model/info`, `POST /predict`, `POST /predict/batch`.

**Output contract** (matches `serving_config.json`): `late_probability` is
the primary output; `predicted_late` is `late_probability >= decision_threshold`
(currently `0.6205`, the F1-optimal threshold tuned on validation) and is a
policy knob that can change without retraining. The score is **not** a
calibrated probability — don't present it as one.

**Input contract**: the API rejects any of the post-outcome fields in
`feature_config.json`'s `api_forbidden_fields` (delivery dates, reviews,
`order_approved_at`, `approval_delay_hours`) — these were never available
at the checkout-time prediction point Notebook 5 targets, and must not be
allowed to influence a live prediction.

## Run the tests

```bash
pytest
```

40/40 passing, verified against your real artifacts — including actual
`/predict` calls that return real predictions and exercise both the
critical-rejection and warning-flag paths (see item 4 below), plus a test
that hides the local model file entirely and confirms the registry path
(item 5) still works, not mocks.

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "order_purchase_timestamp": "2026-01-01T10:00:00",
  "order_estimated_delivery_date": "2026-01-15T00:00:00",
  "total_freight_value": 20.0, "total_item_value": 100.0, "total_payment_value": 120.0,
  "payment_count": 1, "max_payment_installments": 1,
  "item_count": 1, "unique_sellers": 1, "unique_products": 1,
  "customer_zip_code_prefix": "01001", "customer_city": "sao paulo", "customer_state": "SP",
  "seller_zip_code_prefix": "20000", "seller_city": "rio de janeiro", "seller_state": "RJ",
  "distance_km": 430.0
}'
# -> {"late_probability": 0.6066, "predicted_late": false,
#     "model_version": "notebook06-histgbm-checkout-v2", "data_quality_warnings": []}
```

## Data & artifact versioning — DVC (item 4)

`models/artifacts/` is tracked as **one DVC bundle**, not five independent
files. That's deliberate: the model, preprocessor, and target encoder were
all produced by the same training run and only make correct predictions
together — versioning them separately would let you accidentally mix an
old encoder with a new model and get silently wrong (not erroring)
predictions. `models/artifacts.dvc` holds the hash for the whole bundle;
that file (not the actual artifacts) is what gets committed to git.

A local remote is already configured in `.dvc/config` (resolves to
`dvc-storage/` at the repo root, git-ignored) purely so this scaffold is
self-contained and testable without any cloud account. **Point it at real
storage before using this with a team**:

```bash
dvc remote add -d storage s3://your-bucket/olist-artifacts   # or gs://, azure://, etc.
dvc remote modify storage --local ...                        # credentials, not committed
```

Day-to-day workflow:

```bash
dvc pull                       # fetch models/artifacts/ from the remote
# ...retrain, artifacts change...
dvc add models/artifacts       # recompute the hash
git add models/artifacts.dvc
git commit -m "Retrain: <what changed>"
dvc push                       # push the new artifact bundle to the remote
```

I verified this round-trips correctly: deleted `models/artifacts/` and the
local DVC cache entirely, ran `dvc pull`, and got back byte-identical files
(same MD5) from the configured remote.

`data/` doesn't have anything DVC-tracked yet — your raw/processed data
currently lives in local PostgreSQL (`olist_db`), and DVC tracks files, not
live DB connections. Export your Notebook 3 train/val/test splits into
`data/` and `dvc add` them the same way once you want those versioned too.

## Data quality validation — Great Expectations (item 4)

Two layers, in order:

1. **`src/validation.py`** — cheap, fast-fail checks: are the forbidden
   (post-outcome) fields absent, are all required fields present. Runs
   first so obviously-wrong requests never reach the heavier check.
2. **`src/expectations.py`** — the Great Expectations suite
   (`great_expectations/expectations/order_payload_suite.json`, built by
   `scripts/build_expectation_suite.py`): column types, ranges, logical
   consistency (e.g. `item_count >= unique_products`), and allowed
   categories.

**Reject vs. flag, decided per expectation, not globally** — this is
Task 3 item 4's "decide what the service does when validation fails":

- `severity="critical"` — structurally invalid (a state that isn't a real
  Brazilian state, negative distance, an estimated delivery date before
  the purchase date). Raises `ValidationError` → API returns `400`. No
  prediction happens.
- `severity="warning"` — valid but statistically unusual (a real state the
  model rarely or never saw in training, an unusually high order value).
  Logged and returned in the response's `data_quality_warnings` — the
  request still gets a prediction, because rejecting a state the model
  can still score via its infrequent-category bucket would be overly
  strict.
- **`"default"` (silently substituting a value) is deliberately not
  used anywhere** — there's no field where inventing a value is safer
  than rejecting or flagging it. That's a decision, not an oversight.

The range bounds in `scripts/build_expectation_suite.py` are generous
placeholder sanity caps — the notebooks didn't hand me real percentiles,
so these catch genuinely malformed input (a 10,000-item order) without
pretending to be statistically tight. Once you have real p99s from
Notebook 4's EDA, update `SOFT_UPPER_BOUNDS` and rerun:

```bash
python scripts/build_expectation_suite.py
```

One tradeoff worth knowing: the GE check adds real latency (~20–30ms
measured per request here) on top of the model itself — worth it for a
training exercise emphasizing correctness, but if this were a very
high-QPS service you'd profile whether to keep it synchronous per-request
or move it to batch/sampled validation instead.

## Experiment tracking & model registry — MLflow (item 5)

`scripts/log_to_mlflow.py` logs the **real** Notebook 6 run — actual
hyperparameters and validation/test metrics from `models/results_summary.json`,
not placeholders — then registers the model and points a `"champion"`
alias at the new version:

```bash
python scripts/log_to_mlflow.py
# Run ID: eecfac0b676c42a8829140359d17cf37
# Registered olist_late_delivery version 1
# Alias 'champion' -> version 1
```

**The service loads the model from this registry, not a local file**
(`src/model_registry.py`, wired into `src/predict.py`'s `get_model()`):
it resolves `models:/olist_late_delivery@champion` at load time. I proved
this is real, not just configured: a test hides `final_model.joblib`
entirely and confirms the service still loads and predicts correctly —
it can only be coming from the registry.

Promoting a retrained model to serve traffic means re-running the script
(which moves the alias to the new version) — no code change, no redeploy.
Rolling back is `client.set_registered_model_alias(name, "champion",
<previous_version>)`.

**What's registry vs. what's still DVC**: only the model itself is
registered in MLflow. The four preprocessing artifacts (preprocessor,
target encoder, feature names/config) stay DVC-tracked as the item-4
bundle and load locally — they're the feature pipeline, not "the model,"
and DVC already versions them. The logging script does still attach them
as MLflow run artifacts (under `preprocessing/`) purely for provenance —
so from the registry you can always trace which run produced the model
*and* its matching preprocessing objects, even though the runtime service
doesn't fetch them from there.

**Fallback, not hard dependency**: if the registry is unreachable or
nothing's registered yet, `get_model()` logs a warning and falls back to
the local DVC-tracked `final_model.joblib` rather than crashing the
service. Check `/model/info`'s `registry` field to see which path is
actually in effect — it reports `null` plus a `registry_error` if the
fallback triggered.

A local SQLite file (`mlflow.db`, git-ignored) backs the registry here so
this scaffold works standalone — its actual model artifact bytes land in
a local `mlruns/` folder (also git-ignored) alongside it, the sqlite-mode
equivalent of the `mlflow-artifacts` Docker volume in item 8. That's a
dev/demo convenience, not a team setup — a per-developer local file
defeats the point of a *shared* registry. Point `MLFLOW_TRACKING_URI` (or
`config.yaml`'s `mlflow.tracking_uri`) at a real MLflow server before more
than one person needs to see the same registered models.

## Docker & Docker Compose (item 8)

**Honest caveat up front**: I don't have a Docker daemon in my execution
environment, so I couldn't run `docker build` or `docker compose up`
myself — everything else in this README was verified by actually running
it; this section wasn't, in that specific sense. What I *could* do, and
did: validated `docker-compose.yml` against the real Compose Specification
schema (not just YAML syntax), and — the part most likely to actually
break — proved the Postgres-backed MLflow HTTP-server pattern for real:
started an actual `mlflow server` against a real Postgres backend
(outside Docker), pointed `scripts/log_to_mlflow.py` and
`src/model_registry.py` at it over HTTP, and confirmed logging,
registration, and loading all work exactly as the compose stack expects.
I also ran the exact production entrypoint (`uvicorn app.main:app`) with
*only* the runtime dependencies from `requirements.txt` installed in an
isolated venv — simulating the container's Python environment — and
confirmed `/health`, `/model/info`, and a real `/predict` call all work.
Please run `docker compose up --build` on your machine before relying on
this for anything real.

**One command on a clean machine:**

```bash
cp .env.example .env    # fill in real Postgres credentials
dvc pull                 # populate models/artifacts/ (item 4)
docker compose up --build
```

Startup order is enforced by healthchecks, not guesswork: `postgres`
healthy → `mlflow` healthy → `mlflow-init` logs the real training run and
registers the model (exits 0) → `api` starts. No manual step in between.

**"The database" in this stack is Postgres backing the MLflow tracking
server** — item 5's registry needs a real database, not the local SQLite
file used for standalone dev — not a copy of the raw Olist data
warehouse. This service never queries a database at request time; a
single order's fields arrive directly in the API payload.

**Images**: the root `Dockerfile` is a two-stage build (deps installed
into a venv in a builder stage, copied into a slim final image) for the
API — no notebooks, no tests, no dev tools, runs as a non-root user.
`docker/mlflow.Dockerfile` is a separate, smaller image with just
`mlflow` + a Postgres driver — the tracking server doesn't need
`fastapi`/`sklearn`/`pandas`. `mlflow-init` reuses the API image (it needs
the same `joblib`/`mlflow`/`sklearn` to load and log the model) with its
command overridden — pragmatic reuse rather than a third near-identical
image for one narrow job.

**What's baked in vs. mounted**: `models/` (the DVC-tracked preprocessing
bundle *and* `results_summary.json`) is a volume mount, not copied into
the image — retraining means updating the mount, not rebuilding. The
model itself never touches the image or the mount; it's registered into
MLflow's own artifact storage (the `mlflow-artifacts` volume), which is
the literal "storage for the artifacts" item 8 asks for.

**Secrets**: `.env.example` documents the required variables
(`POSTGRES_USER`/`PASSWORD`/`DB`) with placeholder values; `.env` itself
is git-ignored. `docker-compose.yml` only ever references `${VARS}`.

**Known simplification**: `mlflow-init` re-runs and registers a new model
version on *every* `docker compose up`, not just when something changed —
fine for a training exercise, but a real system would gate this (compare
a checksum of `results_summary.json`/artifacts against the currently
registered version, or only run this step during an explicit release,
not on every container start).

## CI/CD (item 9)

`.github/workflows/ci.yml` runs on every push and PR to `main`: lint
(`ruff`) → format check (`black`) → `dvc pull` → register the model for
this run (`scripts/log_to_mlflow.py`, local SQLite — no Postgres/server
needed for CI) → `pytest`. The image only builds and pushes to
`ghcr.io/<repo>` after all of that passes, and only on a push to `main` —
never on a PR.

**A failing test really does stop the pipeline**, not just log a warning:
`build-and-push` declares `needs: lint-and-test`, and GitHub Actions skips
a job outright when a job it `needs` fails. I proved every step in that
sequence works, in order, from a clean slate — deleted
`models/artifacts/`, the DVC cache, and the local MLflow state entirely,
then ran `dvc pull` → `python scripts/log_to_mlflow.py` → `ruff check .`
→ `black --check .` → `pytest`, all green. I also hit the one real gap
firsthand: I'd cleaned up my own local DVC remote for packaging hygiene,
and `dvc pull` failed exactly as the workflow's own comments warn it
will — confirming that warning is accurate, not just a formality.

**What I could verify vs. what I couldn't**: every individual command
above genuinely ran, in this exact sequence, in my sandbox. I could not
push this to an actual GitHub repo or watch a real Actions run — no
GitHub Actions runner available to me here. Before trusting this: push
it and watch the first run.

**The real gap you'll hit**: `dvc pull` in CI needs a real DVC remote
(S3/GCS/Azure) with credentials in repo secrets — the local-filesystem
remote from item 4 is a sibling folder on *my* machine, unreachable from
a GitHub-hosted runner. The workflow has commented-out `AWS_ACCESS_KEY_ID`/
`AWS_SECRET_ACCESS_KEY` env wiring for an S3 remote as a starting point;
swap for GCS/Azure equivalents if that's what you use, and update
`.dvc/config`'s remote URL to match.

**Pre-commit** (`.pre-commit-config.yaml`, install with `pre-commit
install`): the same `ruff`/`black` checks plus basic hygiene
(trailing whitespace, merge-conflict markers, a 1MB file-size cap that
doubles as a backstop against `git add`-ing a model file directly instead
of `dvc add`-ing it) — deliberately fast-only, no test suite, so commits
stay quick; the full suite still runs in CI on every push. I actually ran
`pre-commit run --all-files` against this codebase (not just written the
config and assumed): it downloaded and ran all three hook repos for
real, and caught genuine issues on the first pass — a handful of JSON
files missing a trailing newline, which `end-of-file-fixer` fixed
automatically. `ruff`/`black` were already clean by then because I'd run
them directly first and fixed what they found (a few unsorted imports
and lines over 88 chars) before writing `pyproject.toml`.

## Monitoring (item 10)

**`/metrics`** exposes request count, latency, and error rate
(`api_requests_total`, `api_request_latency_seconds`) via a middleware
that wraps every route automatically — plus a distinct set of
*prediction*-distribution metrics that are model observability, not API
observability: `prediction_late_probability` (a histogram of the actual
scores), `prediction_predicted_late_total` (the thresholded outcome
rate), and `data_quality_warning_total` by column. That last one is
deliberate: a rising rate of "rare state" warnings (item 4) is an early,
specific drift signal — cheaper to act on than waiting for
`prediction_late_probability`'s shape to visibly shift.

I ran a **live Prometheus instance scraping a live copy of this API** to
verify this, not just written the instrumentation and assumed it works:
confirmed the scrape target came up healthy and a real query against
scraped data returned real values matching the requests I'd just made.

**Prediction logs** (`src/prediction_log.py`) write one structured JSON
line per prediction to `logs/predictions.jsonl` — separate from the
operational `app.log` item 3 covers, built specifically so
`scripts/evaluate_predictions.py` can join it against the real delivery
outcome once known (by `order_id`, if the caller sent one — optional,
dropped before it reaches the model, but useful to send for this) and
compute actual accuracy/precision/recall/F1, compared directly against
Notebook 6's own reported test metrics. I tested the join and metric
math against synthetic predictions + ground truth — worked out to the
same numbers by hand.

**Alerting** (`monitoring/alerts.yml`) — decided and written down, not
just implied by having metrics:

| Alert | Severity | Why |
|---|---|---|
| `ServiceDown` | critical | scrape target unreachable |
| `HighServerErrorRate` | critical | >5% of requests 5xx over 5m — the service itself is broken |
| `HighClientErrorRate` | warning | >20% of requests 4xx over 15m — bad input upstream or a client bug, not an outage |
| `HighPredictLatency` | warning | /predict p95 > 1s over 5m — GE alone measured ~20-30ms, so this points elsewhere |
| `PredictedLateRateDrift` | warning | predicted-late rate over a day strays outside 2-25%, vs. training's ~9% baseline |
| `DataQualityWarningSpike` | warning | >10% of an hour's requests carry a GE warning |
| `ModelRegistryFallback` | critical | serving from the local fallback, not the registry — should never happen |

Two severities, deliberately: **critical** means something is broken
right now (page); **warning** means something looks off over hours, not
seconds (drift doesn't page — nobody should be woken up for it).

I validated these with the real Prometheus toolchain, not just eyeballed
YAML: downloaded `promtool` and `prometheus` binaries, ran
`promtool check rules` (7/7 valid) and `promtool check config`, then ran
an actual live Prometheus against a live API and confirmed **all 7 rules
evaluated with `ok` health against real scraped data** — genuine runtime
verification, not just syntax checking.

One real bug caught in the process: `ModelRegistryFallback`'s underlying
metric was originally a Counter incremented once at startup — which
would've been unreliable to alert on, since a single event can roll out
of any `rate()`/`increase()` window depending on when the container
started relative to the query. Fixed it to a Gauge
(`model_using_registry`, checked with a plain `== 0`) before writing the
alert, since it's persistent state for the process's lifetime, not a
repeating event.

**What's not wired up**: no Alertmanager here — `docker-compose.yml`'s
`prometheus` service evaluates the rules and would show firing alerts in
its own UI, but routing them to Slack/PagerDuty is an Alertmanager config
addition, not something worth building for a training exercise. No
statistical drift test (PSI/KS) either — the metrics above make drift
*observable* in Grafana/Prometheus; a fuller system would add scheduled
statistical comparisons against a training-data baseline (`evidently` or
similar) on top of this, not instead of it.

## Configuration

All paths and parameters come from `config/config.yaml`, overridable via
env vars (see `src/config.py`). `model.version` and `model.decision_threshold`
are copied from `serving_config.json` — update them there if you retrain.

## Two real bugs found while wiring this up (already fixed)

1. **`SmoothTargetEncoder` couldn't unpickle outside the notebook.** It was
   defined inline in Notebook 5, so `target_encoder.joblib` was pickled with
   a `__main__.SmoothTargetEncoder` reference — that class doesn't exist in
   any other process (the API, pytest, a plain script). Fixed by moving the
   class into `src/target_encoder.py` and registering it under `__main__`
   at import time in `src/data_access.py`. This is a real production
   footgun, not a test-only issue — it would have broken the live API the
   same way.

2. **scikit-learn version mismatch isn't just a warning here.** The
   artifacts were pickled with **scikit-learn 1.6.1**. A newer install
   (1.8.0, tested) loads `SimpleImputer`/`StandardScaler`/`OneHotEncoder`
   fine but **hard-fails** on `preprocessor.joblib` with
   `AttributeError: Can't get attribute '_RemainderColsList'` — an internal
   `ColumnTransformer` class that changed between versions. `requirements.txt`
   already pins `scikit-learn==1.6.1`; install from it exactly, don't just
   treat the pin as a suggestion.

## What's confirmed

Everything in `src/preprocessing.py` (the `engineer_features` logic, the
target-encoding step, the leakage columns) is ported directly from
Notebook 5's code. The raw input schema in `src/validation.py`'s
`REQUIRED_FIELDS` is now confirmed against Notebook 4's explicit
prediction-time leakage audit (`KNOWN_SAFE` / `KNOWN_POST_OUTCOME` /
`KNOWN_REVIEW`) — the accounting is exact: target (1) + safe (17) +
post-outcome (5) + ID/ambiguous (5) = 28 raw columns, nothing left
unclassified.

## Status

- [x] Repo structure, config-driven paths, split requirements, README
- [x] Logging (console + file, prediction request logging with latency)
- [x] `src/preprocessing.py` — real transform logic ported from Notebook 5
- [x] `src/validation.py` — forbidden-field rejection + completeness check, confirmed against Notebook 4's audit
- [x] FastAPI service with health / model-info / predict / predict-batch routes
- [x] All Notebook 5/6 artifacts wired in — `/predict` returns real predictions
- [x] DVC — `models/artifacts/` versioned as one bundle, round-trip verified (delete + `dvc pull` restored byte-identical files)
- [x] Great Expectations — critical-vs-warning severity split; `data_quality_warnings` surfaced in the API response
- [x] MLflow — real training run logged, model registered with a `"champion"` alias; service loads from the registry, verified with the local file hidden; falls back to DVC-tracked local artifact if the registry is unreachable
- [x] Unit + model + validation + expectations + registry + metrics + prediction-log + integration tests, 40/40 passing
- [x] Docker & Docker Compose — postgres + mlflow + mlflow-init + api, one-command startup; compose file validated against the real Compose Specification schema; core mechanics (Postgres-backed MLflow over HTTP, runtime-only deps) proven outside Docker since no daemon is available here — **not run end-to-end with an actual Docker daemon, please verify on your machine**
- [x] CI/CD — GitHub Actions (lint, format, dvc pull, register, test, build+push on main only); pre-commit hooks actually run and verified (caught real issues on first pass); full pipeline sequence proven from a clean slate — **not run against an actual GitHub repo; you'll need a real DVC remote + secrets for `dvc pull` to succeed in CI, see the item 9 section above**
- [x] Monitoring — `/metrics` (request + prediction-distribution metrics), structured prediction logs for later ground-truth evaluation, 7 alerting rules; verified with a live Prometheus scraping a live API, all rules evaluating `ok` against real data — this is the one item verified as thoroughly as everything else in this project, no caveats

All 10 items done.
