"""
Tests for Task 2.4: behavioural features are computed server-side from stored history.
PayPal mock mode + temporary database.
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
from backend.app.behavioral_detection.anomaly_detector import BehavioralAnomalyDetector

CLEAN_WALLET = "0xde5a010000000000000000000000000000000001"
BUYER_01 = dict(sender_id="user_buyer_01", recipient_id="merchant_tech_supplies",
                device_id="known_macbook_pro_01", recipient_wallet=CLEAN_WALLET,
                sender_country="US", recipient_country="US")


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


def analyze(client, amount=45.0, order_id=None, **fields):
    body = {"transaction_id": f"TX-{uuid.uuid4().hex[:10]}", "amount": amount, "currency": "USD",
            **BUYER_01, **fields}
    if order_id:
        body["order_id"] = order_id
    resp = client.post("/fraud/analyze", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json(), client.get(f"/fraud/score/{body['transaction_id']}").json()


def factors(result):
    return {f["factor"] for f in result["contributing_factors"]}


def profile_of(stored):
    return stored["raw_payload"]["server_computed_profile"]


# --- Client-supplied history is ignored ---------------------------------------

def test_client_average_is_ignored(client):
    """Old bypass: claim a huge historical average so a big payment looks normal."""
    result, stored = analyze(client, amount=4000.0, historical_avg_amount=100000.0, transactions_last_24h=0)
    assert profile_of(stored)["avg_amount"] == 50.0           # from history, not the request
    assert "Transaction Rule: Amount Spike" in factors(result)
    assert "historical_avg_amount" not in stored["raw_payload"]


def test_velocity_counts_server_side_attempts(client):
    for _ in range(6):
        analyze(client, transactions_last_24h=0)               # client claims zero, server counts
    result, stored = analyze(client, transactions_last_24h=0)
    assert profile_of(stored)["attempts_last_24h"] == 6
    assert "Behaviour Rule: High Velocity" in factors(result)


# --- Name tricks no longer work -----------------------------------------------

def test_innocent_looking_recipient_is_still_new(client):
    result, stored = analyze(client, recipient_id="totally_normal_shop")
    assert profile_of(stored)["recipient_seen_before"] is False
    assert "Behaviour Rule: New Recipient" in factors(result)


def test_innocent_looking_device_is_still_unknown(client):
    result, stored = analyze(client, device_id="laptop")
    assert profile_of(stored)["device_seen_before"] is False
    assert "Behaviour Rule: Unrecognized Device" in factors(result)


def test_known_recipient_and_device_are_recognised(client):
    result, stored = analyze(client)
    p = profile_of(stored)
    assert p["recipient_seen_before"] and p["device_seen_before"]
    assert result["risk_band"] == "LOW"


def test_missing_device_is_not_trusted(client):
    result, _ = analyze(client, device_id=None)
    assert "No device identifier" in " ".join(f["description"] for f in result["contributing_factors"])


# --- No history -> UNKNOWN, never LOW -----------------------------------------

def test_new_sender_is_unknown_and_needs_step_up(client):
    result, stored = analyze(client, sender_id="user_brand_new_77")
    assert result["signal_availability"]["behaviour_baseline"] == "UNKNOWN"
    assert result["risk_band"] != "LOW"
    assert any("No payment history" in o for o in result["policy_overrides"])
    assert profile_of(stored)["history_count"] == 0


# --- Only completed payments build the baseline -------------------------------

def run_order(client, sender, recipient, device, amount=45.0, pass_step_up=True):
    order_id = client.post("/payments/create", json={"amount": amount, "recipient_email": "m@example.com"}).json()["order_id"]
    client.post(f"/payments/{order_id}/verify-approval")
    result, _ = analyze(client, amount=amount, order_id=order_id, sender_id=sender,
                        recipient_id=recipient, device_id=device)
    if result["action"] == "STEP_UP" and pass_step_up:
        code = client.post(f"/payments/{order_id}/step-up/start").json()["simulated_code"]
        client.post(f"/payments/{order_id}/step-up/verify", json={"code": code})
    client.post(f"/payments/capture/{order_id}")
    return order_id, result


def test_captured_payment_becomes_history(client, temp_db):
    sender = "user_new_88"
    _, first = run_order(client, sender, "corner_bakery", "phone_A")
    assert first["signal_availability"]["behaviour_baseline"] == "UNKNOWN"

    _, second = run_order(client, sender, "corner_bakery", "phone_A")
    assert second["signal_availability"]["behaviour_baseline"] == "KNOWN"
    assert "Behaviour Rule: New Recipient" not in factors(second)
    assert "Behaviour Rule: Unrecognized Device" not in factors(second)
    assert second["risk_band"] == "LOW"


def test_uncaptured_attempts_do_not_train_the_baseline(client, temp_db):
    sender = "user_new_99"
    # Step-up NOT passed -> never captured -> must not become history.
    run_order(client, sender, "corner_bakery", "phone_B", pass_step_up=False)
    assert temp_db.get_completed_payments(sender) == []
    _, again = run_order(client, sender, "corner_bakery", "phone_B")
    assert again["signal_availability"]["behaviour_baseline"] == "UNKNOWN"


# --- Seed data is labelled -----------------------------------------------------

def test_seed_history_is_marked_synthetic():
    profile = BehavioralAnomalyDetector().build_profile("user_buyer_02")
    assert profile["seed_count"] == 5 and profile["stored_count"] == 0
    assert profile["avg_amount"] == 80.0
