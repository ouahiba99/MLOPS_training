"""
Model tests (Task 3, item 6): the model loads, predicts the right shape,
and produces valid probabilities. Uses the real final_model.joblib —
no mocking.
"""

import numpy as np

from src.data_access import load_joblib_artifact


def test_model_loads_and_reports_expected_feature_count():
    model = load_joblib_artifact("model")
    assert model.n_features_in_ == 52


def test_model_predicts_valid_probabilities():
    model = load_joblib_artifact("model")

    rng = np.random.default_rng(42)
    dummy_features = rng.normal(size=(5, model.n_features_in_)).astype("float32")

    proba = model.predict_proba(dummy_features)
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
