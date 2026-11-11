"""
Tests for the server-side pre-capture risk gate (Task 1.1).
Runs in PayPal mock mode against a temporary SQLite database.
"""

import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.api.payments import capture_block_reason
from backend.app.investigation_agent import task_manager as task_manager_module

SCENARIOS = {
    "LOW": dict(amount=45.0, historical_avg_amount=50.0, sender_id="user_buyer_01",
                recipient_id="merchant_tech_supplies", recipient_wallet="0xde5a010000000000000000000000000000000001",
                device_id="known_macbook_pro_01", sender_country="US", recipient_country="US"),
    "MEDIUM": dict(amount=950.0, historical_avg_amount=80.0, sender_id="user_buyer_02",
                   recipient_id="new_digital_giftcard_store", recipient_wallet=None,
                   device_id="unrecognized_linux_vm_88", sender_country="US", recipient_country="SG"),
    "HIGH": dict(amount=4800.0, historical_avg_amount=120.0, sender_id="user_buyer_03",
                 recipient_id="p2p_crypto_seller_99", recipient_wallet="0xde5a020000000000000000000000000000000001",
                 device_id="vpn_tor_exit_node_12", sender_country="US", recipient_country="KY"),
}


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    # Force PayPal mock mode so tests never hit the real Sandbox.
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    yield test_db
    try:
        path.unlink()
    except OSError:
        pass


@pytest.fixture
def client(temp_db):
    return TestClient(app, headers=MERCHANT_HEADERS)


def create_order(client, amount, approve=True):
    resp = client.post("/payments/create", json={
        "amount": amount, "currency": "USD", "recipient_email": "merchant@example.com"})
    assert resp.status_code == 200, resp.text
    order_id = resp.json()["order_id"]
    if approve:  # mock mode -> simulated buyer approval
        assert client.post(f"/payments/{order_id}/verify-approval").status_code == 200
    return order_id


def analyze(client, order_id, band, **overrides):
    body = {"transaction_id": f"TX-{uuid.uuid4().hex[:10]}", "order_id": order_id,
            "currency": "USD", **SCENARIOS[band], **overrides}
    return client.post("/fraud/analyze", json=body)


def order_with_decision(client, band):
    order_id = create_order(client, SCENARIOS[band]["amount"])
    resp = analyze(client, order_id, band)
    assert resp.status_code == 200, resp.text
    assert resp.json()["risk_band"] == band
    return order_id


# --- Missing / invalid decisions ---------------------------------------------

def test_capture_unknown_order_is_rejected(client):
    assert client.post("/payments/capture/ORDER-DOES-NOT-EXIST").status_code == 404


def test_capture_without_buyer_approval_is_rejected(client, temp_db):
    order_id = create_order(client, 45.0, approve=False)
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 409
    assert temp_db.get_payment_order(order_id)["status"] == "CREATED"


def test_capture_without_risk_decision_is_rejected(client, temp_db):
    order_id = create_order(client, 45.0)
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 409
    assert temp_db.get_payment_order(order_id)["status"] == "APPROVED"


# --- Policy enforcement -------------------------------------------------------

def test_low_risk_order_is_captured(client, temp_db):
    order_id = order_with_decision(client, "LOW")
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "CAPTURED"
    assert temp_db.get_payment_order(order_id)["status"] == "CAPTURED"


def test_high_risk_order_is_blocked(client, temp_db):
    order_id = order_with_decision(client, "HIGH")
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 403
    assert temp_db.get_payment_order(order_id)["status"] == "RISK_EVALUATED"


def test_medium_risk_order_requires_step_up(client, temp_db):
    order_id = order_with_decision(client, "MEDIUM")
    order = temp_db.get_payment_order(order_id)
    assert order["step_up_status"] == "PENDING"
    assert client.post(f"/payments/capture/{order_id}").status_code == 403


def test_client_supplied_score_and_override_are_ignored(client, temp_db):
    """The old bypass: sending a low score or force_capture must not unlock a HIGH order."""
    order_id = order_with_decision(client, "HIGH")
    resp = client.post(f"/payments/capture/{order_id}",
                       json={"order_id": order_id, "risk_score": 0, "force_capture": True})
    assert resp.status_code == 403
    assert temp_db.get_payment_order(order_id)["status"] == "RISK_EVALUATED"


# --- Re-scoring / tampering ---------------------------------------------------

def test_order_cannot_be_rescored(client):
    order_id = order_with_decision(client, "HIGH")
    # Attacker re-submits the same order with "clean" telemetry hoping for LOW.
    resp = analyze(client, order_id, "HIGH", **{k: v for k, v in SCENARIOS["LOW"].items() if k != "amount"})
    assert resp.status_code == 409


def test_amount_must_match_order(client):
    order_id = create_order(client, 4800.0)
    resp = analyze(client, order_id, "LOW")  # LOW scenario claims $45 for a $4800 order
    assert resp.status_code == 400


def test_duplicate_transaction_id_is_rejected(client):
    body = {"transaction_id": "TX-DUPLICATE-1", "currency": "USD", **SCENARIOS["LOW"]}
    assert client.post("/fraud/analyze", json=body).status_code == 200
    assert client.post("/fraud/analyze", json=body).status_code == 409


# --- Idempotency / failure handling -------------------------------------------

def test_duplicate_capture_is_rejected(client):
    order_id = order_with_decision(client, "LOW")
    assert client.post(f"/payments/capture/{order_id}").status_code == 200
    assert client.post(f"/payments/capture/{order_id}").status_code == 409


def test_failed_paypal_capture_is_not_marked_captured(client, temp_db, monkeypatch):
    order_id = order_with_decision(client, "LOW")

    async def failing_capture(_order_id):
        return {"name": "UNPROCESSABLE_ENTITY", "message": "Instrument declined"}

    monkeypatch.setattr(payments.paypal_client, "capture_order", failing_capture)
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 502
    order = temp_db.get_payment_order(order_id)
    assert order["status"] == "CAPTURE_FAILED"
    assert order["last_error"] == "UNPROCESSABLE_ENTITY"


def test_blocked_attempts_are_audited(client, temp_db):
    order_id = order_with_decision(client, "HIGH")
    client.post(f"/payments/capture/{order_id}")
    events = [e for e in temp_db.list_payment_events(limit=100) if e["order_id"] == order_id]
    assert any(e["event_type"] == "PAYMENT.CAPTURE.BLOCKED_BY_RISK_GATE" for e in events)
    audit_actions = {log["action"] for log in temp_db.list_audit_logs(limit=200)}
    assert {"ORDER_CREATED", "RISK_DECISION_ATTACHED"} <= audit_actions


# --- Pure policy function (covers paths wired up in later tasks) -------------

@pytest.mark.parametrize("order,allowed", [
    ({"status": "RISK_EVALUATED", "risk_action": "CAPTURE"}, True),
    ({"status": "RISK_EVALUATED", "risk_action": "STEP_UP", "step_up_status": "PENDING"}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "STEP_UP", "step_up_status": "FAILED"}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "STEP_UP", "step_up_status": "PASSED"}, True),
    ({"status": "RISK_EVALUATED", "risk_action": "STEP_UP", "step_up_status": "FAILED", "analyst_decision": None}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "STEP_UP", "step_up_status": "FAILED", "analyst_decision": "APPROVE"}, True),
    ({"status": "RISK_EVALUATED", "risk_action": "STEP_UP", "step_up_status": "FAILED", "analyst_decision": "REJECT"}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "HOLD", "analyst_decision": None}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "HOLD", "analyst_decision": "REJECT"}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "HOLD", "analyst_decision": "APPROVE"}, True),
    ({"status": "CAPTURE_FAILED", "risk_action": "CAPTURE"}, True),
    ({"status": "CAPTURED", "risk_action": "CAPTURE"}, False),
    ({"status": "CREATED", "risk_action": None}, False),
    ({"status": "APPROVED", "risk_action": None}, False),
    ({"status": "CANCELLED", "risk_action": None}, False),
    ({"status": "RISK_EVALUATED", "risk_action": "SOMETHING_ELSE"}, False),
])
def test_capture_policy(order, allowed):
    assert (capture_block_reason(order) is None) == allowed
