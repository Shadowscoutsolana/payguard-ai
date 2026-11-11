r"""
Trains the PayGuard XGBoost fraud model on the real PaySim dataset.

- Data / features / time split: see ml/preprocessing/features.py (balances excluded: label leak).
- No class re-weighting: predicted probabilities stay close to PaySim's real fraud rate,
  which keeps them interpretable when converted to score points.
- Early stopping on the validation period (steps 401-500); metrics on the test period (501-743).
- The saved model is only replaced after the new one passes validation checks;
  the previous model is kept as risk_model.previous.joblib.

Usage (PowerShell, from the project folder):
    .\.venv\Scripts\python.exe -m ml.training.train_baseline
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import xgboost
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.preprocessing.features import (DEFAULT_PAYSIM_FILE, FEATURE_COLUMNS, LABEL_COLUMN,  # noqa: E402
                                       TRAIN_MAX_STEP, VALID_MAX_STEP, build_live_features,
                                       load_paysim, time_split)
from ml.evaluation.evaluate_model import METRICS_PATH, compute_metrics, print_report  # noqa: E402

MODELS_DIR = ROOT_DIR / "ml" / "models"
MODEL_PATH = MODELS_DIR / "risk_model.joblib"
MIN_TEST_ROC_AUC = 0.70  # sanity gate: the new model must clearly beat chance on unseen data


def train_model() -> dict:
    print(f"Loading PaySim from {DEFAULT_PAYSIM_FILE} ...")
    df = load_paysim()
    train, valid, test = time_split(df)
    print(f"Rows (TRANSFER + CASH_OUT): train {len(train):,} | validation {len(valid):,} | test {len(test):,}")
    print(f"Fraud prevalence: train {train[LABEL_COLUMN].mean():.4%} | test {test[LABEL_COLUMN].mean():.4%}")

    model = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
        eval_metric="aucpr", early_stopping_rounds=30, tree_method="hist",
        n_jobs=-1, random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN],
              eval_set=[(valid[FEATURE_COLUMNS], valid[LABEL_COLUMN])], verbose=False)
    print(f"Best iteration (early stopping on validation): {model.best_iteration}")

    metrics = compute_metrics(model, valid, test)
    print_report(metrics)

    bundle = {
        "model": model,
        "model_type": f"xgboost.XGBClassifier (xgboost {xgboost.__version__})",
        "feature_columns": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "PaySim (Kaggle ealaxi/paysim1)", "licence": "CC BY-SA 4.0",
            "nature": "simulated mobile-money transactions, not PayPal data",
            "rows_used": int(len(df)), "types_used": ["TRANSFER", "CASH_OUT"],
            "excluded": "balance columns (label leakage), step as a feature (no real-clock mapping)",
        },
        "split": {"train_steps": f"1-{TRAIN_MAX_STEP}",
                  "validation_steps": f"{TRAIN_MAX_STEP + 1}-{VALID_MAX_STEP}",
                  "test_steps": f"{VALID_MAX_STEP + 1}-743"},
        "test_metrics": metrics,
    }

    # --- Validation gate before replacing the working model ---
    candidate = MODELS_DIR / "risk_model.candidate.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, candidate)
    reloaded = joblib.load(candidate)
    probe = reloaded["model"].predict_proba(build_live_features(45.0, 3))[:, 1]
    checks = {
        "reloads_from_disk": reloaded["feature_columns"] == FEATURE_COLUMNS,
        "predicts_on_live_features": 0.0 <= float(probe[0]) <= 1.0,
        f"test_roc_auc >= {MIN_TEST_ROC_AUC}": metrics["roc_auc"] >= MIN_TEST_ROC_AUC,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not all(checks.values()):
        candidate.unlink(missing_ok=True)
        raise SystemExit("Validation failed: the existing model was NOT replaced.")

    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, MODELS_DIR / "risk_model.previous.joblib")
    candidate.replace(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH} (previous model backed up as risk_model.previous.joblib)")

    METRICS_PATH.write_text(json.dumps({
        "model": bundle["model_type"], "trained_at": bundle["trained_at"], "dataset": bundle["dataset"],
        "split": bundle["split"], "feature_columns": FEATURE_COLUMNS, "test_metrics": metrics,
    }, indent=2), encoding="utf-8")
    print(f"Metrics saved to {METRICS_PATH}")
    return bundle


if __name__ == "__main__":
    train_model()
