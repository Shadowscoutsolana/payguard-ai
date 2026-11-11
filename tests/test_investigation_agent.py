"""
Tests for Phase 5: evidence minimisation, Gemini output validation, fallback honesty,
background report generation and audit logging. Gemini HTTP responses are faked.
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS

from backend.app.investigation_agent import llm_service
from backend.app.investigation_agent.anonymizer import build_case_evidence
from backend.app.investigation_agent.llm_service import (ANALYST_ACTIONS, InvestigationAgentService,
                                                         validate_report)

RING = "0xde5a020000000000000000000000000000000001"
TX = {
    "transaction_id": "TX-SECRET-123", "sender_id": "user_buyer_03", "recipient_id": "p2p_crypto_seller_99",
    "recipient_wallet": RING, "device_id": "vpn_tor_exit_node_12", "amount": 4800.0, "currency": "USD",
    "risk_score": 100, "risk_band": "HIGH", "action": "HOLD",
    "contributing_factors": [
        {"factor": "Blockchain Rule Gate: Cluster Flag", "impact": 50, "description": "ring"},
        {"factor": "Transaction Rule: Amount Spike", "impact": 15, "description": "Payment amount ($4,800.00) is 40.0x the sender's average."},
        {"factor": "Behaviour Rule: Unrecognized Device", "impact": 15, "description": "Device (simulated ID) not seen before."},
        {"factor": "Behaviour Rule: New Recipient", "impact": 10, "description": "First payment to this recipient."},
    ],
    "score_breakdown": {"model_probability": 0.0666, "model_points": 6.7},
    "model_explanations": [{"label": "Recipient's prior received payments", "value": "0 prior payments",
                            "direction": "increases risk", "shap_value": 2.7}],
    "policy_overrides": [],
}
CLUSTER = {"details": "Possible layering indicator: 5-wallet loop within 4.5 hours.",
           "time_window_start": "2026-10-28T01:10:00+00:00", "time_window_end": "2026-10-28T05:40:00+00:00",
           "provenance": ["synthetic"], "evidence_transfers": [{"from": RING, "to": "0x" + "1" * 40}] * 5}


@pytest.fixture
def pack():
    return build_case_evidence(TX, CLUSTER)


def gemini_report(ids=("E1", "E2", "E3"), **over):
    base = {"executive_summary": "The payment shows a possible layering indicator. Several behaviour rules also fired.",
            "top_red_flags": [{"text": f"Flag {i}", "evidence_ids": [i]} for i in ids],
            "recommended_action": "ESCALATE_ENHANCED_DUE_DILIGENCE", "action_rationale": "Wallet indicator present."}
    return {**base, **over}


# --- Evidence minimisation -----------------------------------------------------

def test_pack_contains_no_identifiers(pack):
    blob = json.dumps(pack)
    for secret in ("TX-SECRET-123", "user_buyer_03", "p2p_crypto_seller_99", RING, "vpn_tor_exit_node_12", "0x" + "1" * 40):
        assert secret not in blob, secret


def test_pack_is_numbered_and_categorised(pack):
    assert [e["id"] for e in pack["evidence"]] == [f"E{i}" for i in range(1, len(pack["evidence"]) + 1)]
    assert pack["evidence"][0]["category"] == "wallet_cluster"
    assert set(pack["categories"]) == {"wallet_cluster", "rule_factor", "ml_model"}
    # the cluster rule factor is not duplicated
    assert sum(e["category"] == "wallet_cluster" for e in pack["evidence"]) == 1
    assert not any("Cluster Flag" in e["text"] for e in pack["evidence"] if e["category"] == "rule_factor")


def test_step_up_escalation_is_evidence():
    p = build_case_evidence({**TX, "risk_band": "MEDIUM"}, None, escalation_reason="3 incorrect codes")
    assert p["evidence"][0] == {"id": "E1", "category": "step_up_escalation",
                                "text": "Step-up verification failed: 3 incorrect codes."}


# --- Output validation -----------------------------------------------------------

def test_valid_report_passes():
    assert validate_report(gemini_report(), ["E1", "E2", "E3", "E4"]) is None


@pytest.mark.parametrize("report,why", [
    (gemini_report(ids=("E1", "E2", "E9")), "unknown"),
    (gemini_report(ids=("E1", "E2")), "expected 3"),
    (gemini_report(executive_summary="One. Two. Three."), "1-2 sentences"),
    (gemini_report(recommended_action="APPROVE"), "allowed list"),
    (gemini_report(action_rationale="The analyst should approve the payment."), "payment decision"),
    (gemini_report(top_red_flags=[{"text": "x", "evidence_ids": []}] * 3), "no evidence"),
])
def test_invalid_reports_are_rejected(report, why):
    assert why in validate_report(report, ["E1", "E2", "E3", "E4"])


def test_fewer_evidence_items_means_fewer_flags():
    assert validate_report(gemini_report(ids=("E1",)), ["E1"]) is None


# --- Fallback is honest -----------------------------------------------------------

def test_fallback_restates_stored_evidence_only(pack):
    svc = InvestigationAgentService(); svc.api_key = None
    r = asyncio.run(svc.generate_investigation_report(pack))
    assert r["llm_status"] == "FALLBACK" and "not configured" in r["fallback_reason"]
    texts = {e["text"] for e in pack["evidence"]}
    assert all(f["text"] in texts for f in r["top_red_flags"])
    blob = json.dumps(r).lower()
    for invented in ("account takeover", "15 minutes", "non-cooperative", "virtual machine", "24h"):
        assert invented not in blob
    assert r["evidence_categories_sent"] == []  # nothing sent to any LLM
    assert validate_report(r, [e["id"] for e in pack["evidence"]]) is None


# --- Gemini path (HTTP faked) ----------------------------------------------------

class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


def fake_gemini(monkeypatch, status=200, report=None, captured=None, statuses=None):
    """statuses: optional list of HTTP codes returned on successive calls."""
    seq = list(statuses) if statuses else None
    calls = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False
        async def post(self, url, json=None, headers=None):
            calls.append(url)
            if captured is not None:
                captured.update(url=url, headers=headers, body=json)
            code = seq.pop(0) if seq else status
            text = __import__("json").dumps(report or gemini_report())
            return FakeResp(code, {"candidates": [{"content": {"parts": [{"text": text}]}}]})

    async def no_sleep(_s):
        return None
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    return calls


def gemini_service():
    svc = InvestigationAgentService(); svc.api_key = "test-key-123"; svc.model = "gemini-3.5-flash"
    return svc


def test_valid_gemini_answer_is_used(monkeypatch, pack):
    captured = {}
    fake_gemini(monkeypatch, captured=captured)
    r = asyncio.run(gemini_service().generate_investigation_report(pack))
    assert r["llm_status"] == "GEMINI" and r["generated_by"] == "Gemini (gemini-3.5-flash)"
    assert r["evidence_categories_sent"] == pack["categories"]
    assert r["recommended_action_label"] == ANALYST_ACTIONS["ESCALATE_ENHANCED_DUE_DILIGENCE"]
    # key in header, never in the URL; prompt contains only the evidence pack
    assert "test-key-123" not in captured["url"] and captured["headers"]["x-goog-api-key"] == "test-key-123"
    assert "user_buyer_03" not in json.dumps(captured["body"]) and RING not in json.dumps(captured["body"])


def test_ungrounded_gemini_answer_falls_back(monkeypatch, pack):
    fake_gemini(monkeypatch, report=gemini_report(ids=("E1", "E2", "E42")))
    r = asyncio.run(gemini_service().generate_investigation_report(pack))
    assert r["llm_status"] == "FALLBACK" and "rejected" in r["fallback_reason"]


def test_gemini_outage_falls_back_after_retries_and_logs_what_was_sent(monkeypatch, pack):
    calls = fake_gemini(monkeypatch, status=503)
    r = asyncio.run(gemini_service().generate_investigation_report(pack))
    assert r["llm_status"] == "FALLBACK" and "HTTP 503 after 3 attempt" in r["fallback_reason"]
    assert len(calls) == 3
    # The evidence WAS transmitted, so the privacy log must say so even though Gemini failed.
    assert r["evidence_categories_sent"] == pack["categories"]


def test_temporary_overload_recovers_on_retry(monkeypatch, pack):
    calls = fake_gemini(monkeypatch, statuses=[503, 200])
    r = asyncio.run(gemini_service().generate_investigation_report(pack))
    assert r["llm_status"] == "GEMINI" and len(calls) == 2


def test_non_retryable_error_is_not_retried(monkeypatch, pack):
    calls = fake_gemini(monkeypatch, status=400)
    r = asyncio.run(gemini_service().generate_investigation_report(pack))
    assert r["llm_status"] == "FALLBACK" and len(calls) == 1


# --- End to end: background reports + audit --------------------------------------

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
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    monkeypatch.setattr(payments.settings, "DEMO_MODE", True)
    yield TestClient(app, headers=MERCHANT_HEADERS), test_db


def test_high_risk_alert_gets_report_and_audit(client):
    c, db = client
    tx_id = f"TX-{uuid.uuid4().hex[:8]}"
    c.post("/fraud/analyze", json={"transaction_id": tx_id, "sender_id": "user_buyer_03", "recipient_id": "p2p_crypto_seller_99",
                                   "amount": 4800.0, "currency": "USD", "recipient_wallet": RING,
                                   "device_id": "vpn_tor_exit_node_12", "sender_country": "US", "recipient_country": "KY"})
    alert = db.get_alert(f"ALERT-{tx_id}")
    assert alert["llm_report"]["llm_status"] == "FALLBACK"           # conftest blocks real Gemini
    assert alert["llm_report"]["evidence"][0]["category"] == "wallet_cluster"
    audit = [l for l in db.list_audit_logs(limit=300) if l["action"] == "AI_REPORT_GENERATED"]
    assert audit and "evidence_categories_sent_to_llm" in audit[0]["payload"]


def test_failed_step_up_escalation_gets_report(client):
    c, db = client
    oid = c.post("/payments/create", json={"amount": 950.0, "recipient_email": "m@example.com"}).json()["order_id"]
    c.post(f"/payments/{oid}/verify-approval")
    tx_id = f"TX-{uuid.uuid4().hex[:8]}"
    c.post("/fraud/analyze", json={"transaction_id": tx_id, "order_id": oid, "sender_id": "user_buyer_02",
                                   "recipient_id": "new_digital_giftcard_store", "amount": 950.0, "currency": "USD",
                                   "device_id": "unrecognized_linux_vm_88", "sender_country": "US", "recipient_country": "SG"})
    code = c.post(f"/payments/{oid}/step-up/start").json()["simulated_code"]
    bad = f"{(int(code) + 1) % 10**6:06d}"
    responses = [c.post(f"/payments/{oid}/step-up/verify", json={"code": bad}) for _ in range(3)]
    assert responses[-1].status_code == 403
    report = db.get_alert(f"ALERT-{tx_id}")["llm_report"]
    assert report["evidence"][0]["category"] == "step_up_escalation"
    assert report["recommended_action"] == "VERIFY_WITH_BUYER"
