"""
Central configuration loader.

Everything the service needs — paths, model name/version, log level, API
host/port — is read from config/config.yaml (or the file pointed to by the
CONFIG_PATH env var). Individual values can be overridden by environment
variables, which is how you'll inject different settings in Docker / CI
without touching the YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    models_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    version: str
    target_column: str
    decision_threshold: float
    score_is_calibrated_probability: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    log_dir: Path


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    registered_model_name: str
    model_alias: str


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    model: ModelConfig
    logging: LoggingConfig
    api: ApiConfig
    mlflow: MlflowConfig


def _env_override(value: str, env_var: str) -> str:
    """Environment variables always win over the YAML file."""
    return os.environ.get(env_var, str(value))


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG_PATH))
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    paths = PathsConfig(
        data_dir=Path(_env_override(raw["paths"]["data_dir"], "APP_DATA_DIR")),
        models_dir=Path(_env_override(raw["paths"]["models_dir"], "APP_MODELS_DIR")),
        artifacts_dir=Path(
            _env_override(raw["paths"]["artifacts_dir"], "APP_ARTIFACTS_DIR")
        ),
    )
    model = ModelConfig(
        name=raw["model"]["name"],
        version=_env_override(raw["model"]["version"], "APP_MODEL_VERSION"),
        target_column=raw["model"]["target_column"],
        decision_threshold=float(
            _env_override(raw["model"]["decision_threshold"], "APP_DECISION_THRESHOLD")
        ),
        score_is_calibrated_probability=raw["model"]["score_is_calibrated_probability"],
    )
    logging_cfg = LoggingConfig(
        level=_env_override(raw["logging"]["level"], "APP_LOG_LEVEL"),
        format=raw["logging"]["format"],
        log_dir=Path(raw["logging"]["log_dir"]),
    )
    api = ApiConfig(
        host=_env_override(raw["api"]["host"], "APP_HOST"),
        port=int(_env_override(raw["api"]["port"], "APP_PORT")),
    )
    mlflow_cfg = MlflowConfig(
        tracking_uri=_env_override(
            raw["mlflow"]["tracking_uri"], "MLFLOW_TRACKING_URI"
        ),
        registered_model_name=_env_override(
            raw["mlflow"]["registered_model_name"], "APP_MLFLOW_MODEL_NAME"
        ),
        model_alias=_env_override(
            raw["mlflow"]["model_alias"], "APP_MLFLOW_MODEL_ALIAS"
        ),
    )
    return Config(
        paths=paths, model=model, logging=logging_cfg, api=api, mlflow=mlflow_cfg
    )


# Loaded once at import time; every module just does `from src.config import config`.
config = load_config()
