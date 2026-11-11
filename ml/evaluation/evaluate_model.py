r"""
Model evaluation on the held-out PaySim TEST period (steps 501-743, never seen in training).

Reports ROC-AUC, PR-AUC, and precision / recall / false-positive rate / confusion matrix at:
  - threshold 0.50, and
  - the threshold that maximised F1 on the VALIDATION period (chosen before looking at test).

Important context printed with every result: the test period's fraud prevalence is higher than
PaySim overall (legitimate volume drops in late simulation hours), and PaySim is simulated
mobile-money data. These numbers do NOT measure real-world PayPal performance.

Usage (PowerShell, from the project folder):
    .\.venv\Scripts\python.exe -m ml.evaluation.evaluate_model
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.preprocessing.features import FEATURE_COLUMNS, LABEL_COLUMN, load_paysim, time_split  # noqa: E402

MODEL_PATH = ROOT_DIR / "ml" / "models" / "risk_model.joblib"
METRICS_PATH = Path(__file__).resolve().parent / "evaluation_results.json"


def _at_threshold(y_true, y_prob, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(fp / (fp + tn), 6) if (fp + tn) else 0.0,
        "confusion_matrix": {"true_negatives": int(tn), "false_positives": int(fp),
                             "false_negatives": int(fn), "true_positives": int(tp)},
    }


def compute_metrics(model, valid, test) -> Dict[str, Any]:
    """Metrics on TEST; the F1-optimal threshold is chosen on VALIDATION only."""
    v_prob = model.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
    prec, rec, thr = precision_recall_curve(valid[LABEL_COLUMN], v_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best_threshold = float(thr[int(np.argmax(f1s))]) if len(thr) else 0.5

    y_true = test[LABEL_COLUMN].to_numpy()
    y_prob = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    return {
        "test_rows": int(len(test)),
        "test_fraud_rows": int(y_true.sum()),
        "test_fraud_prevalence": round(float(y_true.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "at_threshold_0_50": _at_threshold(y_true, y_prob, 0.5),
        "at_validation_f1_threshold": _at_threshold(y_true, y_prob, best_threshold),
    }


def print_report(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("  PAYGUARD AI - XGBoost on PaySim - HELD-OUT TEST PERIOD (steps 501-743)")
    print("=" * 64)
    print(f"  Test rows: {metrics['test_rows']:,} | fraud rows: {metrics['test_fraud_rows']:,} "
          f"| prevalence: {metrics['test_fraud_prevalence'] * 100:.2f}%")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}   PR-AUC: {metrics['pr_auc']:.4f} "
          f"(a random model's PR-AUC = prevalence)")
    for key, label in (("at_threshold_0_50", "Threshold 0.50"),
                       ("at_validation_f1_threshold", "Validation-chosen threshold")):
        m = metrics[key]
        cm = m["confusion_matrix"]
        print(f"  {label} ({m['threshold']}): precision {m['precision']:.4f} | recall {m['recall']:.4f} "
              f"| FPR {m['false_positive_rate'] * 100:.3f}%")
        print(f"      TN {cm['true_negatives']:,} | FP {cm['false_positives']:,} | "
              f"FN {cm['false_negatives']:,} | TP {cm['true_positives']:,}")
    print("  NOTE: PaySim is simulated mobile-money data; not a measure of PayPal performance.")
    print("=" * 64 + "\n")


def evaluate_saved_model(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    bundle = joblib.load(model_path)
    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(f"Model features {bundle['feature_columns']} != pipeline features {FEATURE_COLUMNS}")
    _, valid, test = time_split(load_paysim())
    metrics = compute_metrics(bundle["model"], valid, test)
    print_report(metrics)
    results = {"model": bundle["model_type"], "trained_at": bundle["trained_at"],
               "dataset": bundle["dataset"], "split": bundle["split"],
               "feature_columns": FEATURE_COLUMNS, "test_metrics": metrics}
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Metrics saved to {METRICS_PATH}")
    return results


if __name__ == "__main__":
    evaluate_saved_model()
