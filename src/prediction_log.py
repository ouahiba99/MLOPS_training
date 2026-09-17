"""
Stores one structured record per prediction (Task 3, item 10) — separate
from the free-text app.log that item 3 already covers. app.log is for
operational debugging; this is a queryable dataset built specifically so
scripts/evaluate_predictions.py can join it against the real delivery
outcome once it's known and compute actual accuracy/precision/recall,
not just "the service is up."

Format is JSON Lines (one JSON object per line) — appendable without
rewriting the file, and trivially loadable into pandas with
pd.read_json(path, lines=True) for evaluation.

order_id is captured if the caller happened to send one, purely for this
log — src/validation.py doesn't require it and src/preprocessing.py drops
it (it's an ID column, not a feature; see Notebook 4's leakage audit). No
order_id means no join key later, so it's worth sending if the caller has
one, but its absence doesn't block a prediction.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import config

PREDICTIONS_LOG_PATH = config.logging.log_dir / "predictions.jsonl"


def log_prediction(payload: dict[str, Any], result: dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": payload.get("order_id"),  # optional; see module docstring
        "model_version": result["model_version"],
        "late_probability": result["late_probability"],
        "predicted_late": result["predicted_late"],
        "data_quality_warnings": result.get("data_quality_warnings", []),
        # Filled in later by scripts/evaluate_predictions.py once the real
        # outcome is known — absent here on purpose, not just null, so a
        # naive `if "actual_late" in record` check works to tell the two
        # states apart.
    }
    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_predictions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or PREDICTIONS_LOG_PATH
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
