"""
Loads the registered model from the MLflow Model Registry (Task 3, item 5)
rather than a local file. The service resolves
models:/<registered_model_name>@<model_alias> at load time — promoting a
newly retrained model to serve traffic means moving the alias in the
registry (see scripts/log_to_mlflow.py), not touching this service's code
or its local files.

The four preprocessing artifacts (preprocessor, target encoder, feature
names/config) stay DVC-tracked and locally loaded (src/data_access.py) —
they're the feature pipeline, not "the model" the registry is versioning,
and item 4 already covers their versioning. scripts/log_to_mlflow.py does
still log them as run artifacts alongside the model for provenance, so you
can always trace "which run produced the model AND its matching
preprocessing objects" from the registry, even though the runtime service
doesn't fetch them from there.
"""

import mlflow

from src.config import config


def load_registered_model():
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    model_uri = (
        f"models:/{config.mlflow.registered_model_name}@{config.mlflow.model_alias}"
    )
    return mlflow.sklearn.load_model(model_uri)


def get_champion_version_info() -> dict:
    """What version/run the alias currently points at — for /model/info,
    not the hot path. Re-pointing the alias to a new version is reflected
    here immediately, with no redeploy needed."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        config.mlflow.registered_model_name, config.mlflow.model_alias
    )
    return {
        "registered_model_name": config.mlflow.registered_model_name,
        "alias": config.mlflow.model_alias,
        "version": mv.version,
        "run_id": mv.run_id,
    }
