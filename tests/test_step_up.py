"""
Tests for Task 2.2: simulated OTP step-up verification for MEDIUM-risk orders.
Runs in PayPal mock mode against a temporary database.
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


def order_with_band(client, band):
    order_id = client.post("/payments/create", json={
        "amount": SCENARIOS[band]["amount"], "recipient_email": "m@example.com"}).json()["order_id"]
    client.post(f"/payments/{order_id}/verify-approval")
    resp = client.post("/fraud/analyze", json={"transaction_id": f"TX-{uuid.uuid4().hex[:10]}",
                                               "order_id": order_id, "currency": "USD", **SCENARIOS[band]})
    assert resp.json()["risk_band"] == band, resp.text
    return order_id


def start(client, order_id):
    return client.post(f"/payments/{order_id}/step-up/start")


def verify(client, order_id, code):
    return client.post(f"/payments/{order_id}/step-up/verify", json={"code": code})


def wrong(code):
    return f"{(int(code) + 1) % 10**6:06d}"


# --- Happy path ---------------------------------------------------------------

def test_correct_code_unlocks_capture(client, temp_db):
    order_id = order_with_band(client, "MEDIUM")
    assert client.post(f"/payments/capture/{order_id}").status_code == 403  # blocked before step-up

    challenge = start(client, order_id).json()
    assert challenge["delivery"] == "SIMULATED"
    resp = verify(client, order_id, challenge["simulated_code"])
    assert resp.status_code == 200 and resp.json()["step_up_status"] == "PASSED"
    assert temp_db.get_payment_order(order_id)["step_up_status"] == "PASSED"

    assert client.post(f"/payments/capture/{order_id}").status_code == 200


def test_code_is_stored_only_as_hash(client, temp_db):
    order_id = order_with_band(client, "MEDIUM")
    code = start(client, order_id).json()["simulated_code"]
    stored = temp_db.get_active_step_up_challenge(order_id)["code_hash"]
    assert code not in stored and len(stored) == 64


def test_code_hidden_when_demo_mode_off(client, monkeypatch):
    monkeypatch.setattr(payments.settings, "DEMO_MODE", False)
    order_id = order_with_band(client, "MEDIUM")
    assert "simulated_code" not in start(client, order_id).json()


# --- Failure and escalation ---------------------------------------------------

def test_wrong_code_reports_remaining_attempts(client):
    order_id = order_with_band(client, "MEDIUM")
    code = start(client, order_id).json()["simulated_code"]
    resp = verify(client, order_id, wrong(code))
    assert resp.status_code == 400 and "2 attempt" in resp.json()["detail"]


def test_three_wrong_codes_escalate_to_analyst(client, temp_db):
    order_id = order_with_band(client, "MEDIUM")
    code = start(client, order_id).json()["simulated_code"]
    assert verify(client, order_id, wrong(code)).status_code == 400
    assert verify(client, order_id, wrong(code)).status_code == 400
    assert verify(client, order_id, wrong(code)).status_code == 403

    order = temp_db.get_payment_order(order_id)
    assert order["step_up_status"] == "FAILED"
    alert = temp_db.get_alert(f"ALERT-{order['transaction_id']}")
    assert alert and alert["status"] == "PENDING_REVIEW"
    assert "Step-up verification failed" in alert["reasons"][0]

    # Correct code no longer helps, no new codes, capture refused.
    assert verify(client, order_id, code).status_code == 409
    assert start(client, order_id).status_code == 409
    resp = client.post(f"/payments/capture/{order_id}")
    assert resp.status_code == 403 and "escalated" in resp.json()["detail"]


def test_expired_code_is_rejected_and_new_code_works(client, temp_db):
    order_id = order_with_band(client, "MEDIUM")
    code = start(client, order_id).json()["simulated_code"]
    with temp_db._get_connection() as conn:
        conn.execute("UPDATE step_up_challenges SET expires_at = '2000-01-01T00:00:00+00:00' WHERE order_id = ?",
                     (order_id,))
        conn.commit()
    assert verify(client, order_id, code).status_code == 410
    assert verify(client, order_id, code).status_code == 409  # no active code any more

    new_code = start(client, order_id).json()["simulated_code"]
    assert verify(client, order_id, new_code).status_code == 200


def test_new_code_supersedes_old_one(client):
    order_id = order_with_band(client, "MEDIUM")
    old = start(client, order_id).json()["simulated_code"]
    new = start(client, order_id).json()["simulated_code"]
    if old != new:
        assert verify(client, order_id, old).status_code == 400
    assert verify(client, order_id, new).status_code == 200


def test_too_many_codes_escalate(client, temp_db):
    order_id = order_with_band(client, "MEDIUM")
    for _ in range(3):
        assert start(client, order_id).status_code == 200
    assert start(client, order_id).status_code == 403
    assert temp_db.get_payment_order(order_id)["step_up_status"] == "FAILED"


# --- Only for orders that need it ---------------------------------------------

@pytest.mark.parametrize("band", ["LOW", "HIGH"])
def test_step_up_not_available_for_other_bands(client, band):
    order_id = order_with_band(client, band)
    assert start(client, order_id).status_code == 409


def test_step_up_after_pass_is_rejected(client):
    order_id = order_with_band(client, "MEDIUM")
    code = start(client, order_id).json()["simulated_code"]
    verify(client, order_id, code)
    assert start(client, order_id).status_code == 409


@pytest.mark.parametrize("bad", ["12345", "1234567", "12ab56", ""])
def test_malformed_code_is_rejected(client, bad):
    order_id = order_with_band(client, "MEDIUM")
    start(client, order_id)
    assert verify(client, order_id, bad).status_code == 422
