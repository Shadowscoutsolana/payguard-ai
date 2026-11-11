"""
Phase 6 security tests: access control matrix, fail-closed behaviour, CORS, headers,
error hygiene, input validation and startup configuration checks.
"""

import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings
from backend.app.core import security
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module
from tests.keys import MERCHANT_HEADERS, analyst_headers

RING = "0xde5a020000000000000000000000000000000001"
ANALYZE_BODY = {"transaction_id": "TX-SEC-1", "sender_id": "user_buyer_01", "recipient_id": "merchant_tech_supplies",
                "amount": 45.0, "currency": "USD"}


@pytest.fixture
def client(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    return TestClient(app, raise_server_exceptions=False)  # no default key


# --- Access-control matrix ---------------------------------------------------------

MERCHANT_ONLY = [
    ("POST", "/payments/create", {"amount": 10.0, "recipient_email": "m@example.com"}),
    ("POST", "/payments/ORDER-X/verify-approval", None),
    ("POST", "/payments/ORDER-X/step-up/start", None),
    ("POST", "/payments/ORDER-X/step-up/verify", {"code": "123456"}),
    ("POST", "/payments/capture/ORDER-X", None),
    ("POST", "/fraud/analyze", ANALYZE_BODY),
]
MERCHANT_OR_ANALYST = [("GET", "/payments/ORDER-X", None), ("GET", "/fraud/score/TX-1", None),
                       ("GET", "/alerts", None), ("GET", "/webhooks/paypal/events", None)]
ANALYST_ONLY = [("POST", "/alerts/ALERT-X/review", {"analyst_decision": "REJECT", "analyst_notes": "Reason here."})]
PUBLIC = [("GET", "/", None), ("GET", "/health", None), ("GET", "/wallet-clusters/cluster_ring_904", None),
          ("GET", f"/wallets/ethereum/{RING}/risk", None), ("GET", f"/wallets/ethereum/{RING}/graph", None)]


def call(client, method, path, body, headers=None):
    return client.request(method, path, json=body, headers=headers or {})


@pytest.mark.parametrize("method,path,body", MERCHANT_ONLY + MERCHANT_OR_ANALYST + ANALYST_ONLY)
def test_protected_endpoints_reject_missing_key(client, method, path, body):
    assert call(client, method, path, body).status_code == 401


@pytest.mark.parametrize("method,path,body", MERCHANT_ONLY)
def test_analyst_key_cannot_run_merchant_actions(client, method, path, body):
    assert call(client, method, path, body, analyst_headers()).status_code == 401


@pytest.mark.parametrize("method,path,body", ANALYST_ONLY)
def test_merchant_key_cannot_make_analyst_decisions(client, method, path, body):
    assert call(client, method, path, body, MERCHANT_HEADERS).status_code == 401


@pytest.mark.parametrize("method,path,body", MERCHANT_OR_ANALYST)
def test_read_endpoints_accept_either_role(client, method, path, body):
    assert call(client, method, path, body, MERCHANT_HEADERS).status_code != 401
    assert call(client, method, path, body, analyst_headers()).status_code != 401


@pytest.mark.parametrize("method,path,body", PUBLIC)
def test_public_endpoints_need_no_key(client, method, path, body):
    assert call(client, method, path, body).status_code == 200


def test_step_up_code_cannot_be_requested_anonymously(client):
    """Before Phase 6, anyone knowing an order id could fetch the simulated OTP in demo mode."""
    resp = client.post("/payments/ORDER-SANDBOX-1/step-up/start")
    assert resp.status_code == 401 and "simulated_code" not in resp.text


# --- Fail closed / weak keys --------------------------------------------------------

def test_no_keys_configured_fails_closed(client, monkeypatch):
    monkeypatch.setattr(settings, "MERCHANT_API_KEY", None)
    monkeypatch.setattr(settings, "ANALYST_API_KEYS", None)
    assert client.get("/alerts", headers=MERCHANT_HEADERS).status_code == 503


def test_weak_and_placeholder_keys_are_ignored(monkeypatch):
    monkeypatch.setattr(settings, "MERCHANT_API_KEY", "short")
    monkeypatch.setattr(settings, "ANALYST_API_KEYS", "a:your_analyst_api_key_here,b:tooshort,broken-entry")
    assert security.merchant_key() is None and security.analyst_keys() == {}


def test_identity_resolution():
    assert security.identify(MERCHANT_HEADERS["X-API-Key"]) == {"role": "merchant", "id": "merchant"}
    assert security.identify(analyst_headers("analyst.rahul")["X-API-Key"]) == {"role": "analyst", "id": "analyst.rahul"}
    assert security.identify("nope" * 10) is None and security.identify(None) is None


def test_rejected_key_is_never_logged(client, caplog):
    secret = "my-super-secret-wrong-key-123456789"
    with caplog.at_level(logging.WARNING, logger="payguard.security"):
        client.get("/alerts", headers={"X-API-Key": secret})
    assert "Auth failed" in caplog.text and secret not in caplog.text


# --- CORS, headers, error hygiene -----------------------------------------------------

def test_cors_allows_only_dashboard_origin(client):
    pre = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"}
    good = client.options("/fraud/analyze", headers={"Origin": settings.DASHBOARD_URL, **pre})
    evil = client.options("/fraud/analyze", headers={"Origin": "https://evil.example", **pre})
    assert good.headers.get("access-control-allow-origin") == settings.DASHBOARD_URL
    assert "access-control-allow-origin" not in evil.headers
    assert good.headers.get("access-control-allow-credentials") is None


def test_security_headers_present(client):
    h = client.get("/health").headers
    assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert h["cache-control"] == "no-store"


def test_unhandled_errors_do_not_leak_internals(client, monkeypatch):
    def boom(_cluster_id):
        raise RuntimeError("db password is hunter2")
    monkeypatch.setattr(fraud.cluster_detector, "cluster_report", boom)
    resp = client.get("/wallet-clusters/cluster_ring_904")
    assert resp.status_code == 500 and resp.json() == {"detail": "Internal server error."}
    assert "hunter2" not in resp.text


# --- Input validation ------------------------------------------------------------------

@pytest.mark.parametrize("override", [
    {"sender_country": "USA"}, {"recipient_country": "us"}, {"currency": "usd"}, {"currency": "DOLLARS"},
    {"sender_id": "user one"}, {"recipient_id": "<script>"}, {"amount": 0}, {"amount": 1_000_000},
    {"device_id": "x" * 65}, {"order_id": "../../etc"}, {"recipient_wallet": "0x123"},
])
def test_analyze_rejects_malformed_input(client, override):
    assert client.post("/fraud/analyze", json={**ANALYZE_BODY, **override}, headers=MERCHANT_HEADERS).status_code == 422


@pytest.mark.parametrize("email", ["not-an-email", "a@b", "x" * 300 + "@example.com"])
def test_create_rejects_bad_email(client, email):
    resp = client.post("/payments/create", json={"amount": 10.0, "recipient_email": email}, headers=MERCHANT_HEADERS)
    assert resp.status_code == 422


def test_alert_status_filter_is_validated(client):
    assert client.get("/alerts?status=DROP TABLE", headers=MERCHANT_HEADERS).status_code == 422
    assert client.get("/alerts?status=RESOLVED", headers=MERCHANT_HEADERS).status_code == 200


def test_transaction_id_path_is_validated(client):
    assert client.get("/fraud/score/bad id!", headers=MERCHANT_HEADERS).status_code == 400


# --- Startup configuration checks -------------------------------------------------------

def test_production_with_demo_mode_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    assert any("DEMO_MODE" in p for p in security.startup_security_problems())


def test_non_paypal_api_base_is_flagged(monkeypatch):
    monkeypatch.setattr(settings, "PAYPAL_API_BASE", "https://attacker.example")
    assert any("PAYPAL_API_BASE" in p for p in security.startup_security_problems())


def test_good_configuration_has_no_problems(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", "development")
    monkeypatch.setattr(settings, "PAYPAL_API_BASE", "https://api-m.sandbox.paypal.com")
    assert security.startup_security_problems() == []
