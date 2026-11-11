"""
Unit tests for PayGuard AI Risk Engine and the 3 Core Demo Scenarios.
Verifies decision band outputs, latency targets, and explainable factors.
"""

from backend.app.transaction_intelligence.feature_extractor import TransactionFeatureExtractor
from backend.app.behavioral_detection.anomaly_detector import BehavioralAnomalyDetector
from backend.app.blockchain_intelligence.wallet_profile import WalletProfileService
from backend.app.blockchain_intelligence.graph import BlockchainGraphEngine
from backend.app.blockchain_intelligence.cluster_detection import ClusterDetector
from backend.app.risk_engine.scoring import RiskEngine
from backend.app.schemas import RiskBand, DecisionAction


def test_scenario_a_normal_payment():
    """Scenario A: Known clean merchant payment -> Low Risk -> Auto-Capture."""
    tx_extractor = TransactionFeatureExtractor()
    behavior_detector = BehavioralAnomalyDetector()
    wallet_service = WalletProfileService()
    graph_engine = BlockchainGraphEngine()
    cluster_detector = ClusterDetector(graph_engine, wallet_service)
    risk_engine = RiskEngine()

    payload = {
        "amount": 45.0,
        "historical_avg_amount": 50.0,
        "sender_id": "user_buyer_01",
        "recipient_id": "merchant_tech_supplies",
        "recipient_wallet": "0xde5a010000000000000000000000000000000001",
        "device_id": "known_macbook_pro_01",
        "sender_country": "US",
        "recipient_country": "US",
    }

    profile = behavior_detector.build_profile(payload["sender_id"])  # synthetic demo seed
    tx_f = tx_extractor.extract_features(payload, profile)
    bh_f = behavior_detector.detect_anomalies(payload, profile)
    cl_f = cluster_detector.analyze_cluster(payload["recipient_wallet"])
    result = risk_engine.evaluate(tx_f, bh_f, cl_f)

    assert result["risk_band"] == RiskBand.LOW
    assert result["action"] == DecisionAction.CAPTURE
    assert result["risk_score"] <= 30
    assert result["decision_latency_ms"] < 100.0  # Latency target check


def test_scenario_b_unusual_behavior_step_up():
    """Scenario B: Amount spike + unrecognized device -> Medium Risk -> Step-Up OTP."""
    tx_extractor = TransactionFeatureExtractor()
    behavior_detector = BehavioralAnomalyDetector()
    wallet_service = WalletProfileService()
    graph_engine = BlockchainGraphEngine()
    cluster_detector = ClusterDetector(graph_engine, wallet_service)
    risk_engine = RiskEngine()

    payload = {
        "amount": 950.0,
        "historical_avg_amount": 80.0,
        "sender_id": "user_buyer_02",
        "recipient_id": "new_digital_giftcard_store",
        "recipient_wallet": None,
        "device_id": "unrecognized_linux_vm_88",
        "sender_country": "US",
        "recipient_country": "SG",
    }

    profile = behavior_detector.build_profile(payload["sender_id"])  # synthetic demo seed
    tx_f = tx_extractor.extract_features(payload, profile)
    bh_f = behavior_detector.detect_anomalies(payload, profile)
    cl_f = cluster_detector.analyze_cluster(payload["recipient_wallet"])
    result = risk_engine.evaluate(tx_f, bh_f, cl_f)

    assert result["risk_band"] == RiskBand.MEDIUM
    assert result["action"] == DecisionAction.STEP_UP
    assert 31 <= result["risk_score"] < 70
    assert result["decision_latency_ms"] < 100.0


def test_scenario_c_circular_fraud_ring_hold():
    """Scenario C: Tied to 5-node PYUSD circular ring -> High Risk -> Analyst Hold."""
    tx_extractor = TransactionFeatureExtractor()
    behavior_detector = BehavioralAnomalyDetector()
    wallet_service = WalletProfileService()
    graph_engine = BlockchainGraphEngine()
    cluster_detector = ClusterDetector(graph_engine, wallet_service)
    risk_engine = RiskEngine()

    payload = {
        "amount": 4800.0,
        "historical_avg_amount": 120.0,
        "sender_id": "user_buyer_03",
        "recipient_id": "p2p_crypto_seller_99",
        "recipient_wallet": "0xde5a020000000000000000000000000000000001",
        "device_id": "vpn_tor_exit_node_12",
        "sender_country": "US",
        "recipient_country": "KY",
    }

    profile = behavior_detector.build_profile(payload["sender_id"])  # synthetic demo seed
    tx_f = tx_extractor.extract_features(payload, profile)
    bh_f = behavior_detector.detect_anomalies(payload, profile)
    cl_f = cluster_detector.analyze_cluster(payload["recipient_wallet"])
    result = risk_engine.evaluate(tx_f, bh_f, cl_f)

    assert result["risk_band"] == RiskBand.HIGH
    assert result["action"] == DecisionAction.HOLD
    assert result["risk_score"] >= 70
    assert result["requires_analyst_review"] is True
    assert result["cluster_findings"] is not None
    assert result["cluster_findings"].pattern_type == "CIRCULAR_TRANSFER_CYCLE"
