"""
Layer 4: Risk Engine.

The score is an explicit, auditable sum (every point is accounted for in score_breakdown):

    score = BASE_POINTS
          + model points   : XGBoost fraud probability (trained on PaySim) x 100, capped at MODEL_MAX_POINTS
                             -> explained with real SHAP values (model_explanations)
          + rule points    : fixed policy weights for signals PaySim cannot learn
                             (amount spike vs. sender baseline, unseen device, new recipient,
                              high-risk corridor, high velocity)
          + wallet points  : rule-based PYUSD graph gate (ring / fan-out / verified merchant)
    clipped to 0..100, then mapped to LOW (0-30) / MEDIUM (31-69) / HIGH (70-100).

Model explanations (SHAP) and rule factors are kept separate: rule weights are policy choices,
not learned values, and are never labelled as SHAP.

Policy: a missing critical signal (wallet intelligence, sender baseline, ML model) is UNKNOWN,
and UNKNOWN never yields LOW; the band is raised and the override recorded, the score untouched.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from ..schemas import RiskBand, DecisionAction, ContributingFactor, ClusterFinding

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from ml.preprocessing.features import FEATURE_COLUMNS, build_live_features  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = ROOT_DIR / "ml" / "models" / "risk_model.joblib"

BASE_POINTS = 5
MODEL_MAX_POINTS = 40
RULE_POINTS = {
    "amount_spike": 15,       # amount > 3x the sender's completed-payment average
    "unknown_device": 15,     # device not seen in the sender's completed payments
    "new_recipient": 10,      # sender never completed a payment to this recipient
    "high_risk_corridor": 15,
    "high_velocity": 10,      # > 5 attempts by this sender in the last 24 h
}
AMOUNT_SPIKE_RATIO = 3.0

FEATURE_LABELS = {
    "log_amount": "Payment amount",
    "is_transfer": "Transaction type (P2P transfer)",
    "recipient_prior_count": "Recipient's prior received payments",
}


class RiskEngine:
    def __init__(self, low_threshold: int = 30, high_threshold: int = 70, model_path: Path = MODEL_PATH):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.model = None
        self.model_info: Dict[str, Any] = {}
        self.explainer = None
        self._load_model(model_path)

    def _load_model(self, model_path: Path) -> None:
        """Loads the PaySim XGBoost bundle. Any problem -> model UNAVAILABLE (never a silent fake)."""
        if not Path(model_path).exists():
            logger.warning("Risk model not found at %s; ML signal will be UNAVAILABLE.", model_path)
            return
        try:
            bundle = joblib.load(model_path)
            if not isinstance(bundle, dict) or list(bundle.get("feature_columns", [])) != FEATURE_COLUMNS:
                logger.warning("Risk model at %s has an unexpected format; retrain it.", model_path)
                return
            import shap  # imported lazily: heavy import
            self.model = bundle["model"]
            self.explainer = shap.TreeExplainer(self.model)
            self.model_info = {k: bundle.get(k) for k in ("model_type", "trained_at", "dataset", "split")}
        except Exception:
            logger.exception("Could not load risk model from %s", model_path)
            self.model, self.explainer = None, None

    # ------------------------------------------------------------------ model
    def _model_component(self, tx_features: Dict[str, Any]):
        """Returns (probability, points, explanations) or (None, 0, []) when the model is unavailable."""
        if self.model is None:
            return None, 0.0, []
        X = build_live_features(tx_features.get("amount", 0.0), tx_features.get("recipient_prior_receipts", 0))
        prob = float(self.model.predict_proba(X)[0, 1])
        points = round(min(MODEL_MAX_POINTS, prob * 100.0), 1)

        shap_row = self.explainer.shap_values(X)[0]
        raw_values = {
            "log_amount": f"${tx_features.get('amount', 0.0):,.2f}",
            "is_transfer": "yes",
            "recipient_prior_count": f"{int(tx_features.get('recipient_prior_receipts', 0))} prior payments",
        }
        explanations = sorted((
            {
                "feature": name,
                "label": FEATURE_LABELS[name],
                "value": raw_values[name],
                "shap_value": round(float(val), 4),
                "direction": "increases risk" if val > 0 else "decreases risk",
            } for name, val in zip(FEATURE_COLUMNS, shap_row)
        ), key=lambda e: abs(e["shap_value"]), reverse=True)
        return prob, points, explanations[:3]

    # ------------------------------------------------------------------ rules
    @staticmethod
    def _rule_factors(tx_features: Dict[str, Any], behavior_features: Dict[str, Any]) -> List[ContributingFactor]:
        factors: List[ContributingFactor] = []
        ratio = tx_features.get("amount_deviation_ratio", 1.0)
        if tx_features.get("baseline_known") and ratio > AMOUNT_SPIKE_RATIO:
            factors.append(ContributingFactor(
                factor="Transaction Rule: Amount Spike", impact=RULE_POINTS["amount_spike"],
                description=(f"Payment amount (${tx_features.get('amount', 0.0):,.2f}) is {ratio:.1f}x the sender's "
                             f"average completed payment (${tx_features.get('historical_avg_amount', 0.0):,.2f}).")))
        if behavior_features.get("is_unknown_device"):
            factors.append(ContributingFactor(
                factor="Behaviour Rule: Unrecognized Device", impact=RULE_POINTS["unknown_device"],
                description=("No device identifier supplied for this payment."
                             if not behavior_features.get("device_provided", True)
                             else "Device (simulated ID) not seen in this sender's completed payments.")))
        if behavior_features.get("is_new_recipient"):
            factors.append(ContributingFactor(
                factor="Behaviour Rule: New Recipient", impact=RULE_POINTS["new_recipient"],
                description="First payment from this sender to this recipient (none in completed history)."))
        if tx_features.get("is_high_risk_corridor"):
            factors.append(ContributingFactor(
                factor="Transaction Rule: Corridor Risk", impact=RULE_POINTS["high_risk_corridor"],
                description="Cross-border payment routed through an elevated-risk offshore corridor."))
        if tx_features.get("is_high_velocity"):
            factors.append(ContributingFactor(
                factor="Behaviour Rule: High Velocity", impact=RULE_POINTS["high_velocity"],
                description=(f"{tx_features.get('velocity_last_24h')} payment attempts by this sender in the "
                             "last 24 hours (server-computed).")))
        return factors

    @staticmethod
    def _wallet_factor(cluster_analysis: Dict[str, Any]) -> Optional[ContributingFactor]:
        penalty = float(cluster_analysis.get("cluster_risk_penalty", 0))
        if penalty > 0:
            finding = cluster_analysis.get("findings") or {}
            return ContributingFactor(factor="Blockchain Rule Gate: Cluster Flag", impact=penalty,
                                      description=finding.get("details", "Suspicious on-chain wallet clustering detected."))
        if penalty < 0:
            return ContributingFactor(factor="Blockchain Rule Gate: Allowlisted Merchant", impact=penalty,
                                      description=("Recipient wallet is an allowlisted merchant payout wallet with no "
                                                   "suspicious transfer patterns (demo allowlist, synthetic data)."))
        return None

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, tx_features: Dict[str, Any], behavior_features: Dict[str, Any],
                 cluster_analysis: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        prob, model_points, model_explanations = self._model_component(tx_features)
        rule_factors = self._rule_factors(tx_features, behavior_features)
        wallet_factor = self._wallet_factor(cluster_analysis)
        rule_points = sum(f.impact for f in rule_factors)
        wallet_points = wallet_factor.impact if wallet_factor else 0.0

        raw_total = BASE_POINTS + model_points + rule_points + wallet_points
        final_score = int(max(0, min(100, round(raw_total))))
        score_breakdown = {
            "base_points": BASE_POINTS,
            "model_probability": round(prob, 4) if prob is not None else None,
            "model_points": model_points,
            "rule_points": rule_points,
            "wallet_points": wallet_points,
            "raw_total": round(raw_total, 1),
            "final_score": final_score,
        }

        band = (RiskBand.LOW if final_score <= self.low_threshold
                else RiskBand.MEDIUM if final_score < self.high_threshold else RiskBand.HIGH)

        reasons: List[str] = [f.description for f in rule_factors]
        if wallet_factor and wallet_factor.impact > 0:
            reasons.extend(cluster_analysis.get("reasons", []))
        if prob is not None and model_explanations and model_points >= 10:
            top = model_explanations[0]
            reasons.append(f"Fraud model (PaySim-trained) probability {prob:.1%}; strongest driver: "
                           f"{top['label']} ({top['value']}, {top['direction']}).")

        # Signal availability, reported separately from risk.
        wallet_signal = cluster_analysis.get("wallet_signal", "NOT_PROVIDED")
        baseline_status = behavior_features.get("baseline_status", "UNKNOWN")
        model_status = "AVAILABLE" if prob is not None else "UNAVAILABLE"
        signal_availability = {"wallet_intelligence": wallet_signal, "behaviour_baseline": baseline_status,
                               "ml_model": model_status}
        policy_overrides: List[str] = []
        if band == RiskBand.LOW and wallet_signal != "KNOWN":
            band = RiskBand.MEDIUM
            policy_overrides.append(
                "Wallet intelligence " + ("not provided" if wallet_signal == "NOT_PROVIDED" else "unavailable")
                + " -> treated as UNKNOWN; minimum band raised from LOW to MEDIUM (step-up).")
            reasons.append("Recipient wallet risk is unknown, so the payment cannot be auto-captured.")
        if band == RiskBand.LOW and baseline_status != "KNOWN":
            band = RiskBand.MEDIUM
            policy_overrides.append("No payment history for this sender -> behaviour baseline UNKNOWN; "
                                    "minimum band raised from LOW to MEDIUM (step-up).")
            reasons.append("Sender has no payment history, so the payment cannot be auto-captured.")
        if band == RiskBand.LOW and model_status != "AVAILABLE":
            band = RiskBand.MEDIUM
            policy_overrides.append("Fraud model unavailable -> ML signal UNKNOWN; "
                                    "minimum band raised from LOW to MEDIUM (step-up).")
            reasons.append("Fraud model unavailable, so the payment cannot be auto-captured.")

        if band == RiskBand.LOW:
            action, requires_review = DecisionAction.CAPTURE, False
            if not reasons:
                reasons.append("All signals within the sender's normal pattern.")
        elif band == RiskBand.MEDIUM:
            action, requires_review = DecisionAction.STEP_UP, False
            reasons.append("Requires step-up verification (simulated OTP) before capture.")
        else:
            action, requires_review = DecisionAction.HOLD, True
            reasons.append("High-risk threshold exceeded; held for analyst investigation.")

        contributing_factors = rule_factors + ([wallet_factor] if wallet_factor else [])
        contributing_factors.sort(key=lambda f: abs(f.impact), reverse=True)

        cluster_findings_obj = None
        if cluster_analysis.get("findings"):
            f = cluster_analysis["findings"]
            ev = f.get("evidence") or {}
            cluster_findings_obj = ClusterFinding(
                cluster_id=f.get("cluster_id"), pattern_type=f.get("pattern_type"),
                num_wallets=f.get("num_wallets", 0), details=f.get("details", ""),
                time_window_start=ev.get("time_window_start"), time_window_end=ev.get("time_window_end"),
                provenance=ev.get("provenance", []), evidence_transfers=ev.get("transfers", []))

        return {
            "risk_score": final_score,
            "risk_band": band,
            "action": action,
            "decision_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "reasons": reasons,
            "contributing_factors": contributing_factors,   # rules + wallet gate (actual points added)
            "model_explanations": model_explanations,       # SHAP, top 3
            "score_breakdown": score_breakdown,
            "cluster_findings": cluster_findings_obj,
            "requires_analyst_review": requires_review,
            "signal_availability": signal_availability,
            "policy_overrides": policy_overrides,
        }
