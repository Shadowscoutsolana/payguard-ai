"""
Phase 8: tests for paths the coverage report showed were never exercised.
External HTTP is faked; nothing is sent to PayPal or Google.
"""

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module
from backend.app.paypal_integration import client as client_module
from backend.app.paypal_integration.client import PayPalClient
from tests.keys import MERCHANT_HEADERS

VALID_WEBHOOK_HEADERS = {
    "paypal-auth-algo": "SHA256withRSA",
    "paypal-cert-url": "https://api.sandbox.paypal.com/v1/notifications/certs/CERT-1",
    "paypal-transmission-id": "tid-1", "paypal-transmission-sig": "sig",
    "paypal-transmission-time": "2026-11-09T10:00:00Z",
}


# ===================================================================== PayPal client (real-mode HTTP, faked)
class Resp:
    def __init__(self, status, data):
        self.status_code, self._d = status, data

    def json(self):
        return self._d


def fake_http(monkeypatch, routes, calls):
    """routes: {url_suffix: (status, json) or Exception}."""
    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *e): return False

        async def _handle(self, method, url, **kw):
            calls.append({"method": method, "url": url, **kw})
            for suffix, outcome in routes.items():
                if url.endswith(suffix):
                    if isinstance(outcome, Exception):
                        raise outcome
                    return Resp(*outcome)
            raise AssertionError(f"unexpected URL {url}")

        async def post(self, url, **kw): return await self._handle("POST", url, **kw)
        async def get(self, url, **kw): return await self._handle("GET", url, **kw)
    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeClient)


def real_client(webhook_id="WH-REAL-ID"):
    c = PayPalClient()
    c.is_configured, c.client_id, c.client_secret, c.webhook_id = True, "cid", "secret", webhook_id
    return c


def test_access_token_success_and_failure(monkeypatch):
    calls = []
    fake_http(monkeypatch, {"/v1/oauth2/token": (200, {"access_token": "tok-123"})}, calls)
    assert asyncio.run(real_client().get_access_token()) == "tok-123"
    assert calls[0]["auth"] == ("cid", "secret") and calls[0]["data"] == {"grant_type": "client_credentials"}
    fake_http(monkeypatch, {"/v1/oauth2/token": (401, {"error": "invalid_client"})}, [])
    assert asyncio.run(real_client().get_access_token()) is None


def test_get_order_returns_http_status(monkeypatch):
    fake_http(monkeypatch, {"/v1/oauth2/token": (200, {"access_token": "tok"}),
                            "/v2/checkout/orders/ORDER1": (404, {"name": "RESOURCE_NOT_FOUND"})}, [])
    data = asyncio.run(real_client().get_order("ORDER1"))
    assert data["_http_status"] == 404 and data["name"] == "RESOURCE_NOT_FOUND"


@pytest.mark.parametrize("outcome,expected", [
    ((200, {"verification_status": "SUCCESS"}), "SUCCESS"),
    ((200, {"verification_status": "FAILURE"}), "FAILURE"),
    ((400, {"name": "VALIDATION_ERROR"}), "FAILURE"),
    (ConnectionError("network down: internal-host-10.0.0.5"), "FAILURE"),
])
def test_webhook_postback_interprets_paypal_answer(monkeypatch, outcome, expected):
    calls = []
    fake_http(monkeypatch, {"/v1/oauth2/token": (200, {"access_token": "tok"}),
                            "/v1/notifications/verify-webhook-signature": outcome}, calls)
    event = {"id": "WH-1", "event_type": "PAYMENT.CAPTURE.COMPLETED"}
    result = asyncio.run(real_client().verify_webhook_signature(VALID_WEBHOOK_HEADERS, event))
    assert result["verification_status"] == expected
    assert "internal-host" not in json.dumps(result)  # no internal error text leaks
    post = [c for c in calls if c["url"].endswith("verify-webhook-signature")]
    if post:
        body = post[0]["json"]
        assert body["webhook_id"] == "WH-REAL-ID" and body["webhook_event"] == event
        assert body["transmission_sig"] == "sig" and post[0]["headers"]["Authorization"] == "Bearer tok"


# ===================================================================== API fixture
@pytest.fixture
def env(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    return TestClient(app, headers=MERCHANT_HEADERS, raise_server_exceptions=False), test_db


def low_risk_approved_order(client):
    oid = client.post("/payments/create", json={"amount": 45.0, "recipient_email": "m@example.com"}).json()["order_id"]
    client.post(f"/payments/{oid}/verify-approval")
    client.post("/fraud/analyze", json={
        "transaction_id": f"TX-{uuid.uuid4().hex[:8]}", "order_id": oid, "amount": 45.0, "currency": "USD",
        "sender_id": "user_buyer_01", "recipient_id": "merchant_tech_supplies",
        "recipient_wallet": "0xde5a010000000000000000000000000000000001", "device_id": "known_macbook_pro_01"})
    return oid


# ===================================================================== PayPal failures during create / capture
def test_create_when_paypal_errors_returns_502_and_stores_nothing(env, monkeypatch):
    client, db = env
    async def boom(**kw): raise ConnectionError("paypal unreachable")
    monkeypatch.setattr(payments.paypal_client, "create_order", boom)
    resp = client.post("/payments/create", json={"amount": 10.0, "recipient_email": "m@example.com"})
    assert resp.status_code == 502 and "unreachable" not in resp.text


def test_create_when_paypal_returns_no_id_is_502(env, monkeypatch):
    client, _ = env
    async def no_id(**kw): return {"name": "UNPROCESSABLE_ENTITY"}
    monkeypatch.setattr(payments.paypal_client, "create_order", no_id)
    assert client.post("/payments/create", json={"amount": 10.0, "recipient_email": "m@example.com"}).status_code == 502


def test_capture_network_error_is_recorded_and_retryable(env, monkeypatch):
    client, db = env
    oid = low_risk_approved_order(client)
    original = payments.paypal_client.capture_order
    async def boom(_oid): raise TimeoutError("read timeout")
    monkeypatch.setattr(payments.paypal_client, "capture_order", boom)
    resp = client.post(f"/payments/capture/{oid}")
    assert resp.status_code == 502 and db.get_payment_order(oid)["status"] == "CAPTURE_FAILED"
    assert db.get_payment_order(oid)["last_error"] == "TimeoutError"
    monkeypatch.setattr(payments.paypal_client, "capture_order", original)
    assert client.post(f"/payments/capture/{oid}").status_code == 200


def test_simultaneous_captures_only_one_reaches_paypal(env, monkeypatch):
    """Two capture requests at the same instant: the atomic claim lets exactly one through."""
    client, db = env
    oid = low_risk_approved_order(client)
    paypal_calls = []

    async def slow_capture(order_id):
        paypal_calls.append(order_id)
        await asyncio.sleep(0.05)  # both requests are in flight together
        return {"id": order_id, "status": "COMPLETED", "mock_mode": True}
    monkeypatch.setattr(payments.paypal_client, "capture_order", slow_capture)

    async def attempt():
        try:
            return (await payments.execute_capture(oid, initiated_by="test"))["status"]
        except HTTPException as exc:
            return exc.status_code

    async def race():
        return await asyncio.gather(attempt(), attempt(), attempt())

    results = asyncio.run(race())
    assert sorted(map(str, results)) == ["409", "409", "CAPTURED"]
    assert len(paypal_calls) == 1 and db.get_payment_order(oid)["status"] == "CAPTURED"


# ===================================================================== webhook handler branches
def paypal_says_success(monkeypatch):
    async def ok(headers, body): return {"verification_status": "SUCCESS"}
    monkeypatch.setattr(webhooks.paypal_client, "verify_webhook_signature", ok)


def send(client, event):
    return client.post("/webhooks/paypal", json=event, headers=VALID_WEBHOOK_HEADERS)


def test_approval_webhook_for_foreign_order_is_ignored(env, monkeypatch):
    client, _ = env
    paypal_says_success(monkeypatch)
    resp = send(client, {"id": "WH-A1", "event_type": "CHECKOUT.ORDER.APPROVED", "resource": {"id": "NOT-OURS"}})
    assert resp.json()["status"] == "IGNORED"


def test_approval_webhook_for_already_approved_order(env, monkeypatch):
    client, _ = env
    paypal_says_success(monkeypatch)
    oid = client.post("/payments/create", json={"amount": 10.0, "recipient_email": "m@example.com"}).json()["order_id"]
    client.post(f"/payments/{oid}/verify-approval")
    resp = send(client, {"id": "WH-A2", "event_type": "CHECKOUT.ORDER.APPROVED", "resource": {"id": oid}})
    assert resp.json()["status"] == "PROCESSED" and "already" in resp.json()["note"]


def test_approval_webhook_not_confirmed_by_paypal_fails_safely(env, monkeypatch):
    client, db = env
    paypal_says_success(monkeypatch)
    db.create_payment_order("REALORDER1", amount=10.0, currency="USD", paypal_status="CREATED", mock_mode=False)
    async def not_approved(order_id):
        return {"id": order_id, "_http_status": 200, "status": "PAYER_ACTION_REQUIRED"}
    monkeypatch.setattr(payments.paypal_client, "get_order", not_approved)
    resp = send(client, {"id": "WH-A3", "event_type": "CHECKOUT.ORDER.APPROVED", "resource": {"id": "REALORDER1"}})
    assert resp.json()["status"] == "FAILED" and db.get_payment_order("REALORDER1")["status"] == "CREATED"


def test_refund_for_foreign_order_is_ignored(env, monkeypatch):
    client, _ = env
    paypal_says_success(monkeypatch)
    resp = send(client, {"id": "WH-R1", "event_type": "PAYMENT.CAPTURE.REFUNDED",
                         "resource": {"supplementary_data": {"related_ids": {"order_id": "NOT-OURS"}}}})
    assert resp.json()["status"] == "IGNORED"


def test_oversized_webhook_body_is_rejected(env):
    client, _ = env
    resp = client.post("/webhooks/paypal", content=b"{" + b" " * 300_000 + b"}", headers=VALID_WEBHOOK_HEADERS)
    assert resp.status_code == 413


def test_webhook_processing_crash_is_recorded_not_leaked(env, monkeypatch):
    client, db = env
    paypal_says_success(monkeypatch)
    async def crash(event_type, resource): raise RuntimeError("secret-internal-detail")
    monkeypatch.setattr(webhooks, "_process_event", crash)
    resp = send(client, {"id": "WH-X1", "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {}})
    assert resp.status_code == 200 and resp.json()["status"] == "FAILED"
    assert "secret-internal-detail" not in resp.text
    assert [e for e in db.list_payment_events() if e["event_id"] == "WH-X1"][0]["status"] == "FAILED"


# ===================================================================== AI report generation crash
def test_report_generator_crash_is_visible_and_audited(env, monkeypatch):
    _, db = env
    db.save_alert({"alert_id": "ALERT-CRASH", "transaction_id": "TX-CRASH", "status": "PENDING_REVIEW"})
    async def crash(pack): raise ValueError("model exploded")
    monkeypatch.setattr(task_manager_module.task_manager.service, "generate_investigation_report", crash)
    report = asyncio.run(task_manager_module.task_manager.execute_investigation("ALERT-CRASH", {"evidence": []}))
    assert report["llm_status"] == "FAILED" and report["error"] == "ValueError"
    assert db.get_alert("ALERT-CRASH")["llm_report"]["llm_status"] == "FAILED"
    assert any(a["action"] == "AI_REPORT_FAILED" for a in db.list_audit_for(["ALERT-CRASH"]))


# ===================================================================== ML pipeline (tiny hand-made PaySim file)
def tiny_paysim(tmp_path):
    rows = [  # step, type, amount, nameDest, isFraud
        (1, "PAYMENT", 10.0, "C_A", 0),
        (1, "TRANSFER", 500.0, "C_A", 1),   # same hour as the first receipt -> prior must be 0
        (2, "CASH_OUT", 200.0, "C_A", 0),   # 2 receipts in hour 1 -> prior 2
        (3, "TRANSFER", 50.0, "C_B", 0),    # brand-new recipient -> prior 0
        (401, "CASH_OUT", 70.0, "C_A", 0),  # validation period
        (501, "TRANSFER", 90.0, "C_B", 1),  # test period
    ]
    df = pd.DataFrame(rows, columns=["step", "type", "amount", "nameDest", "isFraud"])
    for col in ("nameOrig", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"):
        df[col] = 0
    path = tmp_path / "tiny_paysim.csv"
    df.to_csv(path, index=False)
    return path


def test_paysim_loader_has_no_future_leakage(tmp_path):
    from ml.preprocessing.features import FEATURE_COLUMNS, load_paysim, time_split
    df = load_paysim(tiny_paysim(tmp_path))
    assert set(df.columns) == {"step", *FEATURE_COLUMNS, "isFraud"}           # no balances
    assert len(df) == 5                                                        # PAYMENT row dropped
    prior = dict(zip(zip(df["step"], df["log_amount"].round(4)), np.expm1(df["recipient_prior_count"]).round()))
    assert prior[(1, round(np.log1p(500.0), 4))] == 0                          # same-hour receipt not counted
    assert prior[(2, round(np.log1p(200.0), 4))] == 2                          # both hour-1 receipts counted
    assert prior[(3, round(np.log1p(50.0), 4))] == 0
    train, valid, test = time_split(df)
    assert train["step"].max() <= 400 and valid["step"].tolist() == [401] and test["step"].tolist() == [501]


def test_missing_paysim_file_gives_clear_instructions(tmp_path):
    from ml.preprocessing.features import load_paysim
    with pytest.raises(FileNotFoundError, match="kaggle.com/datasets/ealaxi/paysim1"):
        load_paysim(tmp_path / "nope.csv")


def test_metrics_threshold_is_chosen_on_validation_not_test():
    from ml.evaluation.evaluate_model import compute_metrics

    class Fixed:  # predicts the value stored in log_amount as the probability
        def predict_proba(self, X):
            p = X["log_amount"].to_numpy()
            return np.column_stack([1 - p, p])
    valid = pd.DataFrame({"log_amount": [0.9, 0.8, 0.1, 0.2], "is_transfer": 1, "recipient_prior_count": 0,
                          "isFraud": [1, 1, 0, 0]})
    test = pd.DataFrame({"log_amount": [0.95, 0.3, 0.05, 0.85], "is_transfer": 1, "recipient_prior_count": 0,
                         "isFraud": [1, 0, 0, 0]})
    m = compute_metrics(Fixed(), valid, test)
    assert m["test_rows"] == 4 and m["test_fraud_rows"] == 1
    assert m["at_validation_f1_threshold"]["threshold"] == 0.8          # from validation data only
    cm = m["at_validation_f1_threshold"]["confusion_matrix"]
    assert (cm["true_positives"], cm["false_positives"]) == (1, 1)        # 0.95 TP, 0.85 FP on test
