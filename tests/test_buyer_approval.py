"""
Tests for buyer approval verification (Task 1.3).
Mock-mode orders use simulated approval; "real-mode" orders have PayPal's GET order
response substituted so we can test every server-side check without network calls.
"""

import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pytest
from fastapi.testclient import TestClient

from tests.keys import MERCHANT_HEADERS

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module


@pytest.fixture
def temp_db(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    yield test_db
    try:
        path.unlink()
    except OSError:
        pass


@pytest.fixture
def client(temp_db):
    return TestClient(app, headers=MERCHANT_HEADERS)


def redirect_params(resp):
    assert resp.status_code == 303
    return {k: v[0] for k, v in parse_qs(urlparse(resp.headers["location"]).query).items()}


# --- Mock mode (simulated approval) -----------------------------------------

def test_mock_approval_link_points_to_our_return_url(client):
    resp = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"})
    assert "/payments/return?token=" in resp.json()["approval_url"]


def test_mock_return_url_approves_and_redirects_to_dashboard(client, temp_db):
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    resp = client.get(f"/payments/return?token={order_id}", follow_redirects=False)
    params = redirect_params(resp)
    assert params == {"order_id": order_id, "approval": "APPROVED"}
    order = temp_db.get_payment_order(order_id)
    assert order["status"] == "APPROVED"
    assert order["approval_source"] == "SIMULATED"


def test_risk_analysis_refused_before_approval(client):
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    resp = client.post("/fraud/analyze", json={
        "transaction_id": "TX-PREAPPROVAL", "order_id": order_id, "sender_id": "user_buyer_01",
        "recipient_id": "merchant_tech_supplies", "amount": 45.0, "currency": "USD"})
    assert resp.status_code == 409
    assert "not approved" in resp.json()["detail"]


def test_cancelled_order_cannot_be_approved_or_captured(client, temp_db):
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    assert redirect_params(client.get(f"/payments/cancel?token={order_id}", follow_redirects=False))["approval"] == "CANCELLED"
    assert temp_db.get_payment_order(order_id)["status"] == "CANCELLED"
    assert client.post(f"/payments/{order_id}/verify-approval").status_code == 409
    assert client.post(f"/payments/capture/{order_id}").status_code == 409


def test_approval_is_idempotent(client):
    order_id = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    assert client.post(f"/payments/{order_id}/verify-approval").json()["already_approved"] is False
    assert client.post(f"/payments/{order_id}/verify-approval").json()["already_approved"] is True


def test_malformed_order_ids_are_rejected(client):
    params = redirect_params(client.get("/payments/return?token=bad$id/../x", follow_redirects=False))
    assert params["approval"] == "FAILED" and params["order_id"] == "invalid"
    assert client.post("/payments/capture/bad$id").status_code == 400


# --- Real mode (PayPal GET order substituted) --------------------------------

@pytest.fixture
def real_order(temp_db):
    temp_db.create_payment_order("5O190127TN364715T", amount=45.0, currency="USD",
                                 paypal_status="CREATED", mock_mode=False)
    return "5O190127TN364715T"


def fake_get_order(monkeypatch, response):
    async def _get_order(order_id):
        return {"id": order_id, "_http_status": 200, **response}
    monkeypatch.setattr(payments.paypal_client, "get_order", _get_order)


APPROVED = {"status": "APPROVED", "payer": {"payer_id": "PAYER123"},
            "purchase_units": [{"amount": {"currency_code": "USD", "value": "45.00"}}]}


def test_real_approval_verified_with_paypal(client, temp_db, real_order, monkeypatch):
    fake_get_order(monkeypatch, APPROVED)
    resp = client.post(f"/payments/{real_order}/verify-approval")
    assert resp.status_code == 200, resp.text
    order = temp_db.get_payment_order(real_order)
    assert order["status"] == "APPROVED"
    assert order["approval_source"] == "PAYPAL_API"
    assert order["payer_id"] == "PAYER123"


def test_forged_return_url_without_paypal_approval_is_rejected(client, temp_db, real_order, monkeypatch):
    """Someone visits /payments/return?token=... directly, but PayPal says the buyer never approved."""
    fake_get_order(monkeypatch, {**APPROVED, "status": "PAYER_ACTION_REQUIRED"})
    params = redirect_params(client.get(f"/payments/return?token={real_order}", follow_redirects=False))
    assert params["approval"] == "FAILED"
    assert temp_db.get_payment_order(real_order)["status"] == "CREATED"


def test_amount_mismatch_with_paypal_is_rejected(client, temp_db, real_order, monkeypatch):
    fake_get_order(monkeypatch, {**APPROVED, "purchase_units": [{"amount": {"currency_code": "USD", "value": "4800.00"}}]})
    assert client.post(f"/payments/{real_order}/verify-approval").status_code == 409
    assert temp_db.get_payment_order(real_order)["status"] == "CREATED"


def test_paypal_unknown_order_is_rejected(client, temp_db, real_order, monkeypatch):
    async def _not_found(order_id):
        return {"name": "RESOURCE_NOT_FOUND", "_http_status": 404}
    monkeypatch.setattr(payments.paypal_client, "get_order", _not_found)
    assert client.post(f"/payments/{real_order}/verify-approval").status_code == 502
    assert temp_db.get_payment_order(real_order)["status"] == "CREATED"
