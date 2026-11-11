"""
Tests for Task 2.1: wallet availability vs. wallet risk, and "unknown is never LOW".
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS

from backend.app.blockchain_intelligence.graph import BlockchainGraphEngine
from backend.app.blockchain_intelligence.wallet_profile import WalletProfileService
from backend.app.blockchain_intelligence.cluster_detection import ClusterDetector
from backend.app.transaction_intelligence.feature_extractor import TransactionFeatureExtractor
from backend.app.behavioral_detection.anomaly_detector import BehavioralAnomalyDetector
from backend.app.risk_engine.scoring import RiskEngine
from backend.app.schemas import RiskBand, DecisionAction

CLEAN_MERCHANT = "0xde5a010000000000000000000000000000000001"
RING_MEMBER = "0xde5a020000000000000000000000000000000001"
UNSEEN_ADDRESS = "0x9999999999999999999999999999999999999999"

LOW_RISK_PAYLOAD = dict(amount=45.0, historical_avg_amount=50.0, sender_id="user_buyer_01",
                        recipient_id="merchant_tech_supplies", device_id="known_macbook_pro_01",
                        sender_country="US", recipient_country="US")


@pytest.fixture(scope="module")
def detector():
    return ClusterDetector(BlockchainGraphEngine(), WalletProfileService())


def evaluate(detector, wallet, **overrides):
    payload = {**LOW_RISK_PAYLOAD, "recipient_wallet": wallet, **overrides}
    detector_b = BehavioralAnomalyDetector()
    profile = detector_b.build_profile(payload["sender_id"])
    return RiskEngine().evaluate(
        TransactionFeatureExtractor().extract_features(payload, profile),
        detector_b.detect_anomalies(payload, profile),
        detector.analyze_cluster(wallet),
    )


# --- Lookup: exact match only -----------------------------------------------

@pytest.mark.parametrize("fragment", ["0x", "a", "0x71C", "a190", "..."])
def test_partial_address_never_matches_known_wallet(detector, fragment):
    analysis = detector.analyze_cluster(fragment)
    assert analysis["wallet_signal"] == "UNKNOWN"
    assert analysis["cluster_risk_penalty"] == 0  # the old bug gave -10 here


def test_exact_match_is_case_insensitive(detector):
    analysis = detector.analyze_cluster(CLEAN_MERCHANT.upper())
    assert analysis["wallet_signal"] == "KNOWN"
    assert analysis["cluster_risk_penalty"] < 0


def test_unseen_address_is_unknown_not_clean():
    profile = WalletProfileService().get_wallet_profile(UNSEEN_ADDRESS)
    assert profile["intelligence_status"] == "UNKNOWN"
    assert profile["is_flagged"] is None
    assert "clean" not in profile["label"].lower()


@pytest.mark.parametrize("wallet", [None, "", "   "])
def test_no_wallet_is_not_provided(detector, wallet):
    assert detector.analyze_cluster(wallet)["wallet_signal"] == "NOT_PROVIDED"


# --- Risk policy: unknown is never LOW ---------------------------------------

@pytest.mark.parametrize("wallet,signal", [(None, "NOT_PROVIDED"), (UNSEEN_ADDRESS, "UNKNOWN"), ("0x", "UNKNOWN")])
def test_low_score_with_unknown_wallet_requires_step_up(detector, wallet, signal):
    result = evaluate(detector, wallet)
    assert result["risk_score"] <= 30            # score itself is not inflated
    assert result["risk_band"] == RiskBand.MEDIUM
    assert result["action"] == DecisionAction.STEP_UP
    assert result["signal_availability"] == {"wallet_intelligence": signal, "behaviour_baseline": "KNOWN", "ml_model": "AVAILABLE"}
    assert len(result["policy_overrides"]) == 1


def test_known_clean_wallet_stays_low(detector):
    result = evaluate(detector, CLEAN_MERCHANT)
    assert result["risk_band"] == RiskBand.LOW
    assert result["action"] == DecisionAction.CAPTURE
    assert result["policy_overrides"] == []


def test_override_never_lowers_a_high_band(detector):
    result = evaluate(detector, RING_MEMBER, amount=4800.0, historical_avg_amount=120.0,
                      device_id="vpn_tor_exit_node_12", recipient_country="KY")
    assert result["risk_band"] == RiskBand.HIGH
    assert result["policy_overrides"] == []


# --- API exposes and stores availability + overrides -------------------------

def test_api_returns_and_stores_signal_availability(monkeypatch):
    from backend.app.main import app
    from backend.app.database import Database
    from backend.app.api import payments, fraud, webhooks
    from backend.app.investigation_agent import task_manager as tm

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, tm):
        monkeypatch.setattr(module, "db", test_db)

    client = TestClient(app, headers=MERCHANT_HEADERS)
    resp = client.post("/fraud/analyze", json={"transaction_id": "TX-WALLET-1", "currency": "USD",
                                                **LOW_RISK_PAYLOAD, "recipient_wallet": None})
    body = resp.json()
    assert body["risk_band"] == "MEDIUM"
    assert body["signal_availability"]["wallet_intelligence"] == "NOT_PROVIDED"
    assert body["policy_overrides"]
    stored = client.get("/fraud/score/TX-WALLET-1").json()
    assert stored["policy_overrides"] == body["policy_overrides"]
    try:
        path.unlink()
    except OSError:
        pass  # Windows may still hold the SQLite file open; it is a temp file.
