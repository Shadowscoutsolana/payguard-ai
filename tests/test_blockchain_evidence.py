"""
Tests for Phase 4: computed wallet-cluster indicators, evidence, provenance and the cluster API.
"""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS

from backend.app.blockchain_intelligence.graph import BlockchainGraphEngine, load_blockchain_data
from backend.app.blockchain_intelligence.wallet_profile import WalletProfileService
from backend.app.blockchain_intelligence.cluster_detection import ClusterDetector

RING = "0xde5a020000000000000000000000000000000001"
FAN = "0xde5a030000000000000000000000000000000002"
LEAF = "0xde5a030000000000000000000000000000000003"
MERCHANT = "0xde5a010000000000000000000000000000000001"


def a(i):  # synthetic test address
    return f"0xde5a7e{'0' * 30}{i:04x}"


def detector_for(wallets, edges, allowlisted=()):
    """Builds a detector from custom in-memory data (no files)."""
    data = {"wallets": {w: {"address": w, "label": w, "provenance": "synthetic", "cluster_id": "c_test",
                            "allowlisted_merchant": w in allowlisted} for w in wallets},
            "transfers": [{"transfer_id": f"T{i}", "from": f, "to": t, "amount_pyusd": 100.0, "timestamp": ts,
                           "tx_hash": None, "provenance": "synthetic", "cluster_id": "c_test"}
                          for i, (f, t, ts) in enumerate(edges)],
            "clusters": {"c_test": {"cluster_id": "c_test", "description": "test", "target_wallet": wallets[0],
                                    "members": list(wallets), "provenance": "synthetic"}}}
    return ClusterDetector(BlockchainGraphEngine(data), WalletProfileService(data))


@pytest.fixture(scope="module")
def demo():
    return ClusterDetector(BlockchainGraphEngine(), WalletProfileService())


# --- Indicators are computed, with evidence ------------------------------------

def test_fan_out_is_computed_with_funding_context(demo):
    f = demo.analyze_cluster(FAN)["findings"]
    assert f["pattern_type"] == "RAPID_FAN_OUT"
    ev = f["evidence"]
    assert ev["window_hours"] <= 24
    assert [t["transfer_id"] for t in ev["transfers"]][0] == "SYN-FAN-001"  # the funding inflow
    assert len({t["to"] for t in ev["transfers"][1:]}) == 4
    assert ev["provenance"] == ["synthetic"]


def test_dispersion_receivers_are_not_flagged(demo):
    """They only received money; the old label-based code flagged them anyway."""
    assert demo.analyze_cluster(LEAF)["has_cluster_flag"] is False


def test_slow_cycle_outside_window_is_not_flagged():
    w = [a(1), a(2), a(3)]
    det = detector_for(w, [(w[0], w[1], "2026-10-01T00:00:00Z"), (w[1], w[2], "2026-10-03T00:00:00Z"),
                           (w[2], w[0], "2026-10-06T00:00:00Z")])  # 5 days > 72 h
    assert det.analyze_cluster(w[0])["has_cluster_flag"] is False


def test_fast_cycle_is_flagged():
    w = [a(1), a(2), a(3)]
    det = detector_for(w, [(w[0], w[1], "2026-10-01T00:00:00Z"), (w[1], w[2], "2026-10-01T05:00:00Z"),
                           (w[2], w[0], "2026-10-01T09:00:00Z")])
    f = det.analyze_cluster(w[0])["findings"]
    assert f["pattern_type"] == "CIRCULAR_TRANSFER_CYCLE" and f["evidence"]["window_hours"] == 9


def detector_with_amounts(wallets, edges):
    """edges: (from, to, timestamp, amount)."""
    det = detector_for(wallets, [(f, t, ts) for f, t, ts, _ in edges])
    for tr, (_, _, _, amt) in zip(det.graph_engine.data["transfers"], edges):
        tr["amount_pyusd"] = amt
    return det


def test_pass_through_dispersion_is_flagged():
    w = [a(i) for i in range(0, 6)]  # w[0] funder, w[1] mule, w[2..5] receivers
    edges = [(w[0], w[1], "2026-10-01T00:00:00Z", 4000.0)] + \
            [(w[1], w[i], f"2026-10-01T0{i}:00:00Z", 950.0) for i in range(2, 6)]
    f = detector_with_amounts(w, edges).analyze_cluster(w[1])["findings"]
    assert f["pattern_type"] == "RAPID_FAN_OUT"
    assert f["evidence"]["pass_through_ratio"] == 0.95 and f["evidence"]["funding_sources"] == 1


def test_busy_service_wallet_is_not_fan_out():
    """Real-data lesson: many payouts, many depositors, most funds retained -> not a mule pattern."""
    w = [a(i) for i in range(0, 12)]
    svc = w[0]
    edges = [(w[i], svc, f"2026-10-01T00:0{i}:00Z", 3000.0) for i in range(1, 6)] + \
            [(svc, w[i], f"2026-10-01T01:0{i - 6}:00Z", 200.0) for i in range(6, 12)]
    assert detector_with_amounts(w, edges).analyze_cluster(svc)["has_cluster_flag"] is False


def test_fan_out_without_any_inflow_is_not_flagged():
    w = [a(i) for i in range(1, 6)]
    edges = [(w[0], w[i], f"2026-10-01T0{i}:00:00Z", 100.0) for i in range(1, 5)]
    assert detector_with_amounts(w, edges).analyze_cluster(w[0])["has_cluster_flag"] is False


def test_three_recipients_is_not_fan_out():
    w = [a(1), a(2), a(3), a(4)]
    det = detector_for(w, [(w[0], w[i], f"2026-10-01T0{i}:00:00Z") for i in (1, 2, 3)])
    assert det.analyze_cluster(w[0])["has_cluster_flag"] is False


def test_fan_out_spread_over_days_is_not_flagged():
    w = [a(i) for i in range(1, 6)]
    det = detector_for(w, [(w[0], w[i], f"2026-10-0{i}T00:00:00Z") for i in range(1, 5)])
    assert det.analyze_cluster(w[0])["has_cluster_flag"] is False


def test_allowlisted_merchant_with_suspicious_pattern_gets_penalty_not_credit():
    w = [a(1), a(2), a(3)]
    det = detector_for(w, [(w[0], w[1], "2026-10-01T00:00:00Z"), (w[1], w[2], "2026-10-01T01:00:00Z"),
                           (w[2], w[0], "2026-10-01T02:00:00Z")], allowlisted={w[0]})
    assert det.analyze_cluster(w[0])["cluster_risk_penalty"] == 50


def test_merchant_gets_allowlist_credit_only_when_clean(demo):
    r = demo.analyze_cluster(MERCHANT)
    assert r["cluster_risk_penalty"] == -10 and r["findings"] is None


# --- Real public-chain data is labelled as such --------------------------------

def test_real_transfers_file_is_loaded_with_real_provenance(tmp_path):
    real = tmp_path / "real.json"
    real.write_text(json.dumps({"transfers": [{
        "transfer_id": "0xabc:1", "from": "0x" + "11" * 20, "to": "0x" + "22" * 20, "amount_pyusd": 5.0,
        "timestamp": "2026-10-01T00:00:00Z", "tx_hash": "0x" + "ab" * 32}]}), encoding="utf-8")
    data = load_blockchain_data(demo_path=tmp_path / "none.json", real_path=real)
    assert data["wallets"]["0x" + "11" * 20]["provenance"] == "real_public_chain"
    assert data["transfers"][0]["provenance"] == "real_public_chain"


# --- API ---------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from backend.app.main import app
    from backend.app.database import Database
    from backend.app.api import payments, fraud, webhooks
    from backend.app.investigation_agent import task_manager as tm
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, tm):
        monkeypatch.setattr(module, "db", test_db)
    yield TestClient(app, headers=MERCHANT_HEADERS)


def test_cluster_endpoint_returns_members_transfers_and_evidence(client):
    body = client.get("/wallet-clusters/cluster_ring_904").json()
    assert len(body["members"]) == 5 and len(body["transfers"]) == 5
    assert body["indicators"][0]["pattern_type"] == "CIRCULAR_TRANSFER_CYCLE"
    assert "not proof" in body["disclaimer"]


@pytest.mark.parametrize("cid,code", [("cluster_nope", 404), ("bad id!", 400)])
def test_cluster_endpoint_errors(client, cid, code):
    assert client.get(f"/wallet-clusters/{cid}").status_code == code


def test_analysis_returns_and_stores_evidence(client):
    resp = client.post("/fraud/analyze", json={
        "transaction_id": "TX-EVIDENCE-1", "sender_id": "user_buyer_03", "recipient_id": "p2p_crypto_seller_99",
        "amount": 4800.0, "currency": "USD", "recipient_wallet": RING, "device_id": "vpn_tor_exit_node_12",
        "sender_country": "US", "recipient_country": "KY"})
    cf = resp.json()["cluster_findings"]
    assert cf["time_window_start"] and cf["time_window_end"] and len(cf["evidence_transfers"]) == 5
    alert = client.get("/alerts").json()[0]
    assert len(alert["cluster_findings"]["evidence_transfers"]) == 5


@pytest.mark.parametrize("wallet", ["0x3Fa...88c1", "0x", "0x" + "z" * 40])
def test_malformed_wallet_is_rejected(client, wallet):
    resp = client.post("/fraud/analyze", json={"transaction_id": "TX-BADW", "sender_id": "u", "recipient_id": "r",
                                               "amount": 10.0, "recipient_wallet": wallet})
    assert resp.status_code == 422


@pytest.mark.parametrize("path,code", [
    (f"/wallets/ethereum/{RING}/risk", 200), (f"/wallets/solana/{RING}/risk", 400),
    ("/wallets/ethereum/0x3Fa...88c1/risk", 400), (f"/wallets/ethereum/{RING}/graph?depth=99", 200)])
def test_wallet_endpoints_validate_input(client, path, code):
    assert client.get(path).status_code == code
