"""
Loading previously-fitted artifacts (target encoder, preprocessor, model,
feature names/config) saved by Notebook 5 and Notebook 6. This module never
calls .fit() on anything — only .load().

Filenames below match serving_config.json's "artifacts" block exactly, so
you can copy your artifacts/notebook_05/ and artifacts/notebook_06/ files
straight into models/artifacts/ without renaming anything.
"""

import json
from typing import Any

import joblib

import __main__
from src.config import config
from src.target_encoder import SmoothTargetEncoder

# target_encoder.joblib was pickled from inside the training notebook,
# where SmoothTargetEncoder lived in the notebook's own __main__ module.
# Registering it here means joblib.load() can resolve `__main__.SmoothTargetEncoder`
# no matter what's actually running as __main__ (uvicorn, pytest, a script).
__main__.SmoothTargetEncoder = SmoothTargetEncoder

JOBLIB_ARTIFACTS = {
    "model": "final_model",
    "baseline_model": "baseline_model",
    "preprocessor": "preprocessor",
    "target_encoder": "target_encoder",
}
JSON_ARTIFACTS = {
    "feature_names": "feature_names",
    "feature_config": "feature_config",
}


def load_joblib_artifact(logical_name: str) -> Any:
    filename = JOBLIB_ARTIFACTS[logical_name]
    path = config.paths.artifacts_dir / f"{filename}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — copy it from your Notebook 5/6 artifacts folder "
            f"into {config.paths.artifacts_dir}/"
        )
    return joblib.load(path)


def load_json_artifact(logical_name: str) -> Any:
    filename = JSON_ARTIFACTS[logical_name]
    path = config.paths.artifacts_dir / f"{filename}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — copy it from your Notebook 5 artifacts folder "
            f"into {config.paths.artifacts_dir}/"
        )
    with open(path) as f:
        return json.load(f)


def load_feature_names() -> list[str]:
    """The exact, ordered output of preprocessor.get_feature_names_out()
    from Notebook 5."""
    return load_json_artifact("feature_names")


def load_feature_config() -> dict:
    """Notebook 5's feature_config.json — leakage cols, target-encoded cols,
    OHE cols, and api_forbidden_fields all come from here so nothing is
    hardcoded twice."""
    return load_json_artifact("feature_config")
