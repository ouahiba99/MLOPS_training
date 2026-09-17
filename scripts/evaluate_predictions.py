"""
Task 3, item 10: "Store prediction logs so you can evaluate later when
the real delivery date arrives." This is that evaluation step.

Joins src/prediction_log.py's JSONL log against a ground-truth file
(order_id -> actual_late, exported from olist_db once the real delivery
date is known for those orders) and computes live accuracy/precision/
recall/F1 — then compares them against the ORIGINAL validation/test
metrics in models/results_summary.json. A live F1 that's fallen well
below the ~0.169 test F1 Notebook 6 reported is a concrete, quantified
signal to retrain, not a vague "something feels off."

Ground truth file format: CSV with columns order_id, actual_late (1/0 or
true/false). Predictions with no order_id (the field is optional — see
src/prediction_log.py) can't be joined and are reported separately, not
silently dropped.

Usage:
    python scripts/evaluate_predictions.py path/to/ground_truth.csv
"""

import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.prediction_log import read_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "models" / "results_summary.json"


def load_ground_truth(path: Path) -> dict[str, bool]:
    truth = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            truth[row["order_id"]] = row["actual_late"].strip().lower() in (
                "1",
                "true",
            )
    return truth


def evaluate(predictions: list[dict], ground_truth: dict[str, bool]) -> dict:
    no_order_id = [p for p in predictions if not p.get("order_id")]
    joinable = [p for p in predictions if p.get("order_id") in ground_truth]
    unmatched = [
        p
        for p in predictions
        if p.get("order_id") and p["order_id"] not in ground_truth
    ]

    if not joinable:
        return {
            "n_predictions": len(predictions),
            "n_no_order_id": len(no_order_id),
            "n_unmatched": len(unmatched),
            "n_evaluated": 0,
            "metrics": None,
        }

    y_true = [ground_truth[p["order_id"]] for p in joinable]
    y_pred = [p["predicted_late"] for p in joinable]

    return {
        "n_predictions": len(predictions),
        "n_no_order_id": len(no_order_id),
        "n_unmatched": len(unmatched),
        "n_evaluated": len(joinable),
        "metrics": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        },
    }


def compare_to_baseline(live_metrics: dict) -> None:
    with open(RESULTS_PATH) as f:
        baseline = json.load(f)["final_test_metrics_tuned_threshold"]

    print("\nLive vs. Notebook 6's original test metrics:")
    for key in ["accuracy", "precision", "recall", "f1"]:
        live = live_metrics[key]
        base = baseline[key]
        delta = live - base
        flag = " <-- notably worse, consider retraining" if delta < -0.05 else ""
        print(
            f"  {key:10s}  live={live:.4f}  baseline={base:.4f}  "
            f"delta={delta:+.4f}{flag}"
        )


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate_predictions.py <ground_truth.csv>")
        sys.exit(1)

    ground_truth = load_ground_truth(Path(sys.argv[1]))
    predictions = read_predictions()

    report = evaluate(predictions, ground_truth)
    print(json.dumps(report, indent=2))

    if report["metrics"] is not None:
        compare_to_baseline(report["metrics"])
    else:
        print("\nNo predictions could be matched to ground truth — nothing to compare.")


if __name__ == "__main__":
    main()
