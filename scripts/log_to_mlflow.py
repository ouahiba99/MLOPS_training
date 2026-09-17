"""
Task 3, item 5: experiment tracking & model registry.

Logs the actual Notebook 6 run — real hyperparameters and metrics from
models/results_summary.json, not placeholders — as an MLflow run, then
registers the model and points the "champion" alias at it. The service
(src/model_registry.py) resolves that alias at load time, so promoting a
retrain to serve traffic means re-running this script and moving the
alias, not touching service code.

The four preprocessing artifacts (preprocessor, target encoder, feature
names/config) are logged as run artifacts too — not because the service
loads them from here (it doesn't; see src/model_registry.py's docstring),
but so you can always trace "which run produced this model AND its
matching preprocessing objects" from the registry.

Run this after retraining, once models/artifacts/ and
models/results_summary.json reflect the new run:

    python scripts/log_to_mlflow.py
"""

import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

from src.config import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
RESULTS_PATH = PROJECT_ROOT / "models" / "results_summary.json"


def _parse_final_model_params(raw: str) -> dict:
    # final_model_params is a Python-repr'd dict string in results_summary.json
    # (e.g. "{'learning_rate': 0.1, ...}") — literal_eval, not json.loads.
    import ast

    return ast.literal_eval(raw)


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("olist-late-delivery")

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    client = MlflowClient()

    # Keep Compose startup idempotent: do not register the same model
    # version repeatedly. A new config.model.version represents a new retrain.
    existing_versions = list(
        client.search_model_versions(f"name='{config.mlflow.registered_model_name}'")
    )
    matching_versions = []
    for v in existing_versions:
        if v.tags.get("model_version_name") == config.model.version:
            matching_versions.append(v)
            continue

        if v.run_id:
            run = client.get_run(v.run_id)
            run_name = run.data.tags.get("mlflow.runName")
            if run_name == config.model.version:
                matching_versions.append(v)

    existing = max(
        matching_versions,
        key=lambda v: int(v.version),
        default=None,
    )

    if existing is not None:
        client.set_registered_model_alias(
            config.mlflow.registered_model_name,
            config.mlflow.model_alias,
            existing.version,
        )
        logger.info(
            "Model version %s already registered; alias '%s' -> version %s",
            config.model.version,
            config.mlflow.model_alias,
            existing.version,
        )
        return

    model = joblib.load(config.paths.artifacts_dir / "final_model.joblib")

    with mlflow.start_run(run_name=config.model.version) as run:
        mlflow.log_params(_parse_final_model_params(results["final_model_params"]))
        mlflow.log_params(
            {
                "final_model_family": results["final_model_family"],
                "n_candidates_evaluated": results["n_candidates_evaluated"],
                "primary_metric": results["primary_metric"],
                "optimal_decision_threshold": results["optimal_decision_threshold"],
                "random_state": results["random_state"],
                "n_features": results["n_features"],
            }
        )

        for split, metrics in [
            ("val", results["final_validation_metrics"]),
            ("test", results["final_test_metrics_tuned_threshold"]),
        ]:
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})

        # Preprocessing artifacts — logged for provenance, not for runtime loading.
        for name in [
            "preprocessor.joblib",
            "target_encoder.joblib",
            "feature_names.json",
            "feature_config.json",
        ]:
            mlflow.log_artifact(
                str(config.paths.artifacts_dir / name), artifact_path="preprocessing"
            )

        mlflow.set_tag("model_version_name", config.model.version)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=config.mlflow.registered_model_name,
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )

        client = MlflowClient()
        version = model_info.registered_model_version
        client.set_registered_model_alias(
            config.mlflow.registered_model_name, config.mlflow.model_alias, version
        )

        logger.info("Run ID: %s", run.info.run_id)
        logger.info(
            "Registered %s version %s",
            config.mlflow.registered_model_name,
            version,
        )
        logger.info(
            "Alias '%s' -> version %s",
            config.mlflow.model_alias,
            version,
        )


if __name__ == "__main__":
    main()
