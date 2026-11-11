"""
Tests for Phase 3: PaySim XGBoost model, SHAP explanations and the auditable score.
Model-dependent tests are skipped if ml/models/risk_model.joblib has not been trained yet.
"""

import json
import math
import warnings
from pathlib import Path

import joblib
import pytest

from ml.preprocessing.features import FEATURE_COLUMNS, build_live_features
from backend.app.risk_engine.scoring import MODEL_PATH, RiskEngine, BASE_POINTS, MODEL_MAX_POINTS
from backend.app.schemas import RiskBand

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first (ml.training.train_baseline)")

LOW_TX = {"amount": 45.0, "recipient_prior_receipts": 3, "baseline_known": True, "amount_deviation_ratio": 0.9}
KNOWN_BEHAVIOUR = {"baseline_status": "KNOWN", "is_unknown_device": 0.0, "is_new_recipient": 0.0}
CLEAN_WALLET = {"wallet_signal": "KNOWN", "cluster_risk_penalty": -10}


# --- Feature definition -------------------------------------------------------

def test_no_balance_columns_are_features():
    """PaySim balances leak the label (97.8% of fraud rows: amount == oldbalanceOrg)."""
    assert not any("balance" in c.lower() for c in FEATURE_COLUMNS)
    assert "step" not in FEATURE_COLUMNS


def test_live_features_are_named_and_ordered():
    X = build_live_features(950.0, 0)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.loc[0, "recipient_prior_count"] == 0.0
    assert math.isclose(X.loc[0, "log_amount"], math.log1p(950.0))


# --- Saved model --------------------------------------------------------------

@needs_model
def test_saved_model_is_xgboost_with_matching_features():
    bundle = joblib.load(MODEL_PATH)
    assert type(bundle["model"]).__name__ == "XGBClassifier"
    assert list(bundle["feature_columns"]) == FEATURE_COLUMNS
    assert "PaySim" in bundle["dataset"]["name"]


@needs_model
def test_prediction_has_no_feature_name_warning():
    model = joblib.load(MODEL_PATH)["model"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = model.predict_proba(build_live_features(45.0, 3))[0, 1]
    assert 0.0 <= p <= 1.0
    assert not [w for w in caught if "feature names" in str(w.message)]


@needs_model
def test_evaluation_results_match_saved_model():
    bundle = joblib.load(MODEL_PATH)
    results = json.loads((Path(__file__).resolve().parent.parent / "ml" / "evaluation" /
                          "evaluation_results.json").read_text(encoding="utf-8"))
    assert results["trained_at"] == bundle["trained_at"]
    assert results["test_metrics"]["roc_auc"] == bundle["test_metrics"]["roc_auc"]


# --- SHAP ---------------------------------------------------------------------

@needs_model
def test_shap_values_are_real_and_additive():
    """SHAP values + expected value must reproduce the model's log-odds exactly."""
    engine = RiskEngine()
    X = build_live_features(950.0, 0)
    shap_row = engine.explainer.shap_values(X)[0]
    p = float(engine.model.predict_proba(X)[0, 1])
    log_odds = math.log(p / (1 - p))
    assert math.isclose(float(engine.explainer.expected_value) + float(sum(shap_row)), log_odds, abs_tol=1e-3)


@needs_model
def test_top_three_explanations_with_direction():
    result = RiskEngine().evaluate({"amount": 950.0, "recipient_prior_receipts": 0}, KNOWN_BEHAVIOUR, CLEAN_WALLET)
    exps = result["model_explanations"]
    assert len(exps) == 3
    assert [abs(e["shap_value"]) for e in exps] == sorted((abs(e["shap_value"]) for e in exps), reverse=True)
    assert all(e["direction"] in ("increases risk", "decreases risk") for e in exps)
    assert exps[0]["feature"] == "recipient_prior_count" and exps[0]["direction"] == "increases risk"


@needs_model
def test_rule_factors_are_never_labelled_shap():
    result = RiskEngine().evaluate({**LOW_TX, "amount_deviation_ratio": 12.0, "historical_avg_amount": 80.0},
                                   {**KNOWN_BEHAVIOUR, "is_unknown_device": 1.0}, CLEAN_WALLET)
    assert all("SHAP" not in f.factor and "XGBoost" not in f.factor for f in result["contributing_factors"])


# --- Score breakdown ----------------------------------------------------------

@needs_model
def test_score_breakdown_adds_up():
    result = RiskEngine().evaluate({**LOW_TX, "amount_deviation_ratio": 12.0, "historical_avg_amount": 80.0},
                                   {**KNOWN_BEHAVIOUR, "is_unknown_device": 1.0, "is_new_recipient": 1.0},
                                   {"wallet_signal": "NOT_PROVIDED", "cluster_risk_penalty": 0})
    b = result["score_breakdown"]
    assert b["raw_total"] == pytest.approx(b["base_points"] + b["model_points"] + b["rule_points"] + b["wallet_points"], abs=0.11)
    assert b["rule_points"] == sum(f.impact for f in result["contributing_factors"] if "Blockchain" not in f.factor)
    assert b["final_score"] == result["risk_score"] == max(0, min(100, round(b["raw_total"])))
    assert 0 <= b["model_points"] <= MODEL_MAX_POINTS and b["base_points"] == BASE_POINTS


# --- Model unavailable ---------------------------------------------------------

def test_missing_model_is_unknown_never_low(tmp_path):
    engine = RiskEngine(model_path=tmp_path / "does_not_exist.joblib")
    result = engine.evaluate(LOW_TX, KNOWN_BEHAVIOUR, CLEAN_WALLET)
    assert result["signal_availability"]["ml_model"] == "UNAVAILABLE"
    assert result["score_breakdown"]["model_probability"] is None
    assert result["risk_band"] == RiskBand.MEDIUM
    assert any("model unavailable" in o.lower() for o in result["policy_overrides"])


def test_old_format_model_file_is_rejected_not_used(tmp_path):
    """A bare estimator (the old GradientBoosting format) must not be silently used."""
    from sklearn.dummy import DummyClassifier
    old = tmp_path / "old.joblib"
    joblib.dump(DummyClassifier(), old)
    assert RiskEngine(model_path=old).model is None
