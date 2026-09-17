"""
Task 3, item 10 test coverage: the prediction log.
"""

from src.prediction_log import log_prediction, read_predictions


def test_log_prediction_writes_a_readable_record(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("src.prediction_log.PREDICTIONS_LOG_PATH", log_path)

    payload = {"order_id": "order_123", "item_count": 1}
    result = {
        "late_probability": 0.42,
        "predicted_late": False,
        "model_version": "test-v1",
        "data_quality_warnings": [],
    }
    log_prediction(payload, result)

    records = read_predictions(log_path)
    assert len(records) == 1
    assert records[0]["order_id"] == "order_123"
    assert records[0]["late_probability"] == 0.42
    assert records[0]["predicted_late"] is False
    assert "timestamp" in records[0]


def test_log_prediction_handles_missing_order_id(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("src.prediction_log.PREDICTIONS_LOG_PATH", log_path)

    log_prediction(
        {"item_count": 1},  # no order_id
        {
            "late_probability": 0.1,
            "predicted_late": False,
            "model_version": "test-v1",
            "data_quality_warnings": [],
        },
    )

    records = read_predictions(log_path)
    assert records[0]["order_id"] is None


def test_read_predictions_returns_empty_list_when_file_missing(tmp_path):
    assert read_predictions(tmp_path / "does_not_exist.jsonl") == []


def test_log_prediction_appends_not_overwrites(tmp_path, monkeypatch):
    log_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("src.prediction_log.PREDICTIONS_LOG_PATH", log_path)

    result = {
        "late_probability": 0.1,
        "predicted_late": False,
        "model_version": "test-v1",
        "data_quality_warnings": [],
    }
    log_prediction({"order_id": "a"}, result)
    log_prediction({"order_id": "b"}, result)

    records = read_predictions(log_path)
    assert [r["order_id"] for r in records] == ["a", "b"]
