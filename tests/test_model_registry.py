"""
Task 3, item 5 test coverage: the model registry integration.
"""

from fastapi.testclient import TestClient

from app.main import app
from src.model_registry import get_champion_version_info, load_registered_model

client = TestClient(app)


def test_champion_alias_resolves_to_a_version():
    info = get_champion_version_info()
    assert info["registered_model_name"] == "olist_late_delivery"
    assert info["alias"] == "champion"
    assert info["version"] == 1
    assert info["run_id"]


def test_model_loads_from_registry_with_correct_shape():
    model = load_registered_model()
    assert model.n_features_in_ == 52


def test_model_info_route_surfaces_registry_state():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["registry"] is not None
    assert body["registry"]["alias"] == "champion"
    assert body["registry"]["version"] == 1


def test_get_model_actually_uses_registry_not_local_fallback():
    # Hide the local artifact entirely and force a fresh load — if this
    # still succeeds, the model came from the registry, not the fallback.
    import src.predict as predict_module
    from src.config import config

    local_path = config.paths.artifacts_dir / "final_model.joblib"
    backup_path = config.paths.artifacts_dir / "final_model.joblib.bak"
    predict_module._model = None
    local_path.rename(backup_path)
    try:
        model = predict_module.get_model()
        assert model.n_features_in_ == 52
    finally:
        backup_path.rename(local_path)
        predict_module._model = None  # reset so later tests get a clean load
