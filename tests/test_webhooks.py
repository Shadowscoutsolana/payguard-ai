"""
Tests for PayPal webhook handling (Task 1.5).
PayPal's verification API answer is substituted; nothing is sent over the network.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module
from backend.app.paypal_integration.client import PayPalClient

VALID_HEADERS = {
    "paypal-auth-algo": "SHA256withRSA",
    "paypal-cert-url": "https://api.sandbox.paypal.com/v1/notifications/certs/CERT-1",
    "paypal-transmission-id": "tid-1",
    "paypal-transmission-sig": "sig",
    "paypal-transmission-time": "2026-11-09T10:00:00Z",
}


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)  # mock orders
    yield test_db
    try:
        path.unlink()
    except OSError:
        pass


@pytest.fixture
def client(temp_db):
    return TestClient(app, headers=MERCHANT_HEADERS)


def paypal_says(monkeypatch, status):
    async def _verify(headers, body):
        return {"verification_status": status, "reason": "test"}
    monkeypatch.setattr(webhooks.paypal_client, "verify_webhook_signature", _verify)


def send(client, event):
    return client.post("/webhooks/paypal", json=event, headers=VALID_HEADERS)


def capture_event(event_id, order_id, capture_id, event_type="PAYMENT.CAPTURE.COMPLETED"):
    return {"id": event_id, "event_type": event_type,
            "resource": {"id": capture_id, "status": "COMPLETED",
                         "supplementary_data": {"related_ids": {"order_id": order_id}}}}


def captured_low_risk_order(client):
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    client.post(f"/payments/{order_id}/verify-approval")
    client.post("/fraud/analyze", json={
        "transaction_id": f"TX-{order_id}", "order_id": order_id, "amount": 45.0, "currency": "USD",
        "sender_id": "user_buyer_01", "recipient_id": "merchant_tech_supplies",
        "recipient_wallet": "0xde5a010000000000000000000000000000000001", "device_id": "known_macbook_pro_01", "historical_avg_amount": 50.0})
    assert client.post(f"/payments/capture/{order_id}").status_code == 200
    return order_id


# --- Signature verification (client) ----------------------------------------

def make_client(configured=True, webhook_id="WH-REAL-ID"):
    c = PayPalClient()
    c.is_configured = configured
    c.webhook_id = webhook_id
    return c


def test_mock_mode_never_verifies():
    """The old bypass: headers present + no credentials used to return SUCCESS."""
    result = asyncio.run(make_client(configured=False).verify_webhook_signature(VALID_HEADERS, {"id": "x"}))
    assert result["verification_status"] == "FAILURE"


@pytest.mark.parametrize("webhook_id", [None, "", "your_paypal_webhook_id_here"])
def test_missing_webhook_id_fails(webhook_id):
    result = asyncio.run(make_client(webhook_id=webhook_id).verify_webhook_signature(VALID_HEADERS, {}))
    assert result["verification_status"] == "FAILURE"
    assert "WEBHOOK_ID" in result["reason"]


def test_missing_headers_fail():
    headers = {k: v for k, v in VALID_HEADERS.items() if k != "paypal-transmission-sig"}
    result = asyncio.run(make_client().verify_webhook_signature(headers, {}))
    assert result["verification_status"] == "FAILURE"


@pytest.mark.parametrize("cert_url", [
    "https://evil.example.com/cert", "http://api.paypal.com/cert", "https://api.paypal.com.evil.io/cert"])
def test_untrusted_cert_url_fails(cert_url):
    result = asyncio.run(make_client().verify_webhook_signature({**VALID_HEADERS, "paypal-cert-url": cert_url}, {}))
    assert result["verification_status"] == "FAILURE"
    assert "certificate" in result["reason"]


# --- Receiver endpoint --------------------------------------------------------

def test_forged_event_is_rejected_and_not_stored_under_claimed_id(client, temp_db, monkeypatch):
    paypal_says(monkeypatch, "FAILURE")
    resp = send(client, {"id": "WH-REAL-EVENT-1", "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {}})
    assert resp.status_code == 401
    events = temp_db.list_payment_events()
    assert events[0]["event_id"].startswith("REJECTED-")
    assert not any(e["event_id"] == "WH-REAL-EVENT-1" for e in events)


def test_invalid_json_is_400(client, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    resp = client.post("/webhooks/paypal", content=b"not json", headers=VALID_HEADERS)
    assert resp.status_code == 400


def test_duplicate_event_is_processed_once(client, temp_db, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    order_id = captured_low_risk_order(client)
    order = temp_db.get_payment_order(order_id)
    event = capture_event("WH-DUP-1", order_id, order["capture_id"])
    first, second = send(client, event).json(), send(client, event).json()
    assert first["status"] in ("PROCESSED", "FLAGGED")
    assert second["status"] == "DUPLICATE_IGNORED"
    actions = [l["action"] for l in temp_db.list_audit_logs(limit=200) if l["entity_id"] == "WH-DUP-1"]
    assert actions.count("WEBHOOK_RECEIVED") == 1 and "WEBHOOK_DUPLICATE_IGNORED" in actions


def test_order_approved_webhook_confirms_approval(client, temp_db, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    resp = send(client, {"id": "WH-APP-1", "event_type": "CHECKOUT.ORDER.APPROVED",
                         "resource": {"id": order_id, "status": "APPROVED"}})
    assert resp.json()["status"] == "PROCESSED"
    assert temp_db.get_payment_order(order_id)["status"] == "APPROVED"


def test_capture_outside_risk_gate_is_flagged(client, temp_db, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    resp = send(client, capture_event("WH-OUT-1", order_id, "CAPTURE-XYZ"))
    assert resp.json()["status"] == "FLAGGED"
    assert temp_db.get_payment_order(order_id)["status"] == "CREATED"  # webhook never marks captured
    assert any(l["action"] == "CAPTURE_OUTSIDE_RISK_GATE" for l in temp_db.list_audit_logs(limit=200))


def test_capture_completed_reconciles_with_gated_capture(client, temp_db, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    order_id = captured_low_risk_order(client)
    # Mock captures have no capture id; simulate the real id PayPal would have returned.
    with temp_db._get_connection() as conn:
        conn.execute("UPDATE payment_orders SET capture_id='CAP-1' WHERE order_id=?", (order_id,))
        conn.commit()
    assert send(client, capture_event("WH-REC-1", order_id, "CAP-1")).json()["status"] == "PROCESSED"


def test_refund_updates_paypal_status(client, temp_db, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    order_id = captured_low_risk_order(client)
    resp = send(client, capture_event("WH-REF-1", order_id, "CAP-1", "PAYMENT.CAPTURE.REFUNDED"))
    assert resp.json()["status"] == "PROCESSED"
    assert temp_db.get_payment_order(order_id)["paypal_status"] == "REFUNDED"


def test_unknown_event_type_is_logged_only(client, monkeypatch):
    paypal_says(monkeypatch, "SUCCESS")
    resp = send(client, {"id": "WH-OTHER-1", "event_type": "BILLING.SUBSCRIPTION.CREATED", "resource": {}})
    assert resp.json()["status"] == "IGNORED"
