"""
Tests for Task 2.3: analyst decisions drive real (gated) capture.
PayPal mock mode + temporary database.
"""

import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS, analyst_headers

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module

HIGH = dict(amount=4800.0, historical_avg_amount=120.0, sender_id="user_buyer_03",
            recipient_id="p2p_crypto_seller_99", recipient_wallet="0xde5a020000000000000000000000000000000001",
            device_id="vpn_tor_exit_node_12", sender_country="US", recipient_country="KY")
MEDIUM = dict(amount=950.0, historical_avg_amount=80.0, sender_id="user_buyer_02",
              recipient_id="new_digital_giftcard_store", recipient_wallet=None,
              device_id="unrecognized_linux_vm_88", sender_country="US", recipient_country="SG")


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    monkeypatch.setattr(payments.settings, "DEMO_MODE", True)
    yield test_db
    try:
        path.unlink()
    except OSError:
        pass


@pytest.fixture
def client(temp_db):
    return TestClient(app, headers=MERCHANT_HEADERS)


def held_order(client, scenario=HIGH):
    order_id = client.post("/payments/create", json={"amount": scenario["amount"],
                                                     "recipient_email": "m@example.com"}).json()["order_id"]
    client.post(f"/payments/{order_id}/verify-approval")
    tx_id = f"TX-{uuid.uuid4().hex[:10]}"
    client.post("/fraud/analyze", json={"transaction_id": tx_id, "order_id": order_id,
                                        "currency": "USD", **scenario})
    return order_id, f"ALERT-{tx_id}"


def review(client, alert_id, decision, analyst="analyst.priya", notes="Reviewed wallet evidence."):
    return client.post(f"/alerts/{alert_id}/review", headers=analyst_headers(analyst), json={
        "analyst_decision": decision, "analyst_notes": notes})


# --- APPROVE ------------------------------------------------------------------

def test_approve_captures_through_the_gate(client, temp_db):
    order_id, alert_id = held_order(client)
    assert client.post(f"/payments/capture/{order_id}").status_code == 403  # held before review

    resp = review(client, alert_id, "APPROVE")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["capture"]["status"] == "CAPTURED"
    assert body["capture_status"] == "CAPTURED"

    order = temp_db.get_payment_order(order_id)
    assert order["status"] == "CAPTURED"
    assert order["analyst_decision"] == "APPROVE" and order["analyst_id"] == "analyst.priya"
    assert order["decided_at"]


def test_approve_reports_real_capture_failure(client, temp_db, monkeypatch):
    order_id, alert_id = held_order(client)

    async def declined(_order_id):
        return {"name": "INSTRUMENT_DECLINED", "message": "declined"}
    monkeypatch.setattr(payments.paypal_client, "capture_order", declined)

    body = review(client, alert_id, "APPROVE").json()
    assert body["capture"]["status"] == "CAPTURE_FAILED"
    assert body["capture_status"] == "CAPTURE_FAILED"
    assert temp_db.get_payment_order(order_id)["status"] == "CAPTURE_FAILED"  # never shown as captured


def test_capture_can_be_retried_after_failure_with_analyst_approval(client, temp_db, monkeypatch):
    order_id, alert_id = held_order(client)
    original = payments.paypal_client.capture_order

    async def declined(_order_id):
        return {"name": "INSTRUMENT_DECLINED"}
    monkeypatch.setattr(payments.paypal_client, "capture_order", declined)
    review(client, alert_id, "APPROVE")

    monkeypatch.setattr(payments.paypal_client, "capture_order", original)
    assert client.post(f"/payments/capture/{order_id}").status_code == 200


# --- REJECT -------------------------------------------------------------------

def test_reject_blocks_capture_permanently(client, temp_db):
    order_id, alert_id = held_order(client)
    body = review(client, alert_id, "REJECT", notes="Recipient wallet in circular ring.").json()
    assert body["capture"]["status"] == "BLOCKED_BY_ANALYST"
    assert temp_db.get_payment_order(order_id)["analyst_decision"] == "REJECT"
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 403 and "Analyst rejected" in resp.json()["detail"]


# --- One decision only --------------------------------------------------------

def test_alert_cannot_be_decided_twice(client):
    _, alert_id = held_order(client)
    assert review(client, alert_id, "REJECT").status_code == 200
    resp = review(client, alert_id, "APPROVE", analyst="analyst.rahul")
    assert resp.status_code == 409


def test_unknown_alert_is_404(client):
    assert review(client, "ALERT-NOPE", "APPROVE").status_code == 404


@pytest.mark.parametrize("payload", [
    {"analyst_decision": "APPROVE", "analyst_notes": "ok"},          # reason too short
    {"analyst_decision": "MAYBE", "analyst_notes": "Not sure."},     # not a valid decision
])
def test_review_requires_valid_reason_and_decision(client, payload):
    _, alert_id = held_order(client)
    assert client.post(f"/alerts/{alert_id}/review", json=payload, headers=analyst_headers()).status_code == 422


@pytest.mark.parametrize("headers", [
    {"X-API-Key": ""},                                   # no key
    {"X-API-Key": "test-analyst-priya-0123456789abcX"},  # wrong key
    MERCHANT_HEADERS,                                    # merchant cannot make analyst decisions
])
def test_review_requires_an_analyst_key(client, temp_db, headers):
    order_id, alert_id = held_order(client)
    resp = client.post(f"/alerts/{alert_id}/review", headers=headers,
                       json={"analyst_decision": "APPROVE", "analyst_notes": "Looks fine to me."})
    assert resp.status_code == 401
    assert temp_db.get_alert(alert_id)["status"] == "PENDING_REVIEW"
    assert temp_db.get_payment_order(order_id)["status"] == "RISK_EVALUATED"


def test_identity_comes_from_key_not_body(client, temp_db):
    order_id, alert_id = held_order(client)
    client.post(f"/alerts/{alert_id}/review", headers=analyst_headers("analyst.rahul"),
                json={"analyst_decision": "REJECT", "analyst_notes": "Ring evidence.", "analyst_id": "ceo.override"})
    assert temp_db.get_payment_order(order_id)["analyst_id"] == "analyst.rahul"


# --- Step-up escalation path --------------------------------------------------

def test_failed_step_up_can_be_approved_by_analyst(client, temp_db):
    order_id, alert_id = held_order(client, MEDIUM)
    code = client.post(f"/payments/{order_id}/step-up/start").json()["simulated_code"]
    bad = f"{(int(code) + 1) % 10**6:06d}"
    for _ in range(3):
        client.post(f"/payments/{order_id}/step-up/verify", json={"code": bad})
    assert temp_db.get_payment_order(order_id)["step_up_status"] == "FAILED"

    body = review(client, alert_id, "APPROVE", notes="Called buyer; confirmed purchase.").json()
    assert body["capture"]["status"] == "CAPTURED"


def test_medium_order_without_failed_step_up_has_no_alert(client, temp_db):
    order_id, alert_id = held_order(client, MEDIUM)
    assert temp_db.get_alert(alert_id) is None  # no escalation -> no analyst path


# --- Audit --------------------------------------------------------------------

def test_decision_is_audited_with_identity(client, temp_db):
    order_id, alert_id = held_order(client)
    review(client, alert_id, "APPROVE", analyst="analyst.rahul")
    logs = [l for l in temp_db.list_audit_logs(limit=300) if l["action"] == "ANALYST_APPROVE"]
    assert {l["entity_id"] for l in logs} >= {alert_id, order_id}
    assert all("analyst.rahul" in l["payload"] for l in logs)
