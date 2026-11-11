r"""
LIVE tests against the real external services (PayPal Sandbox, Gemini, Etherscan).

Skipped unless you opt in, so the normal test run never touches external services:
    $env:RUN_LIVE_TESTS = "1"; .\.venv\Scripts\python.exe -m pytest tests\test_live_services.py -v

They use the real credentials in .env, create Sandbox orders that are never approved or captured,
send only synthetic evidence to Gemini, and read public PYUSD data from Etherscan.
"""

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

from backend.app.config import Settings

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1",
                                reason="live external-service tests; set RUN_LIVE_TESTS=1 to run")
REAL = Settings()  # reads .env directly (the conftest test keys do not apply here)


def configured(value):
    return bool(value) and not str(value).startswith("your_")


# ------------------------------------------------------------------ PayPal Sandbox
@pytest.fixture
def paypal():
    if not configured(REAL.PAYPAL_CLIENT_ID):
        pytest.skip("PayPal Sandbox credentials not configured")
    from backend.app.paypal_integration.client import PayPalClient
    client = PayPalClient()
    assert client.is_configured and "sandbox" in client.base_url
    return client


def test_sandbox_token(paypal):
    assert asyncio.run(paypal.get_access_token())


def test_sandbox_order_lifecycle_without_buyer(paypal):
    order = asyncio.run(paypal.create_order(amount=12.34, currency="USD", description="PayGuard live test"))
    assert order.get("id") and order.get("status") == "PAYER_ACTION_REQUIRED"
    assert any(link["rel"] == "payer-action" for link in order["links"])

    fetched = asyncio.run(paypal.get_order(order["id"]))
    assert fetched["_http_status"] == 200 and fetched["intent"] == "CAPTURE"
    assert fetched["purchase_units"][0]["amount"]["value"] == "12.34"

    # PayPal itself must refuse to capture an order the buyer never approved.
    result = asyncio.run(paypal.capture_order(order["id"]))
    assert result.get("status") != "COMPLETED"


def test_sandbox_rejects_forged_webhook(paypal):
    if not configured(REAL.PAYPAL_WEBHOOK_ID):
        pytest.skip("PAYPAL_WEBHOOK_ID not configured (deferred live webhook setup)")
    headers = {"paypal-auth-algo": "SHA256withRSA",
               "paypal-cert-url": "https://api.sandbox.paypal.com/v1/notifications/certs/CERT-360caa42-fca2a594-1d93a270",
               "paypal-transmission-id": "forged-1", "paypal-transmission-sig": "Zm9yZ2Vk",
               "paypal-transmission-time": "2026-11-09T10:00:00Z"}
    result = asyncio.run(paypal.verify_webhook_signature(headers, {"id": "WH-FORGED", "event_type": "X"}))
    assert result["verification_status"] == "FAILURE"


# ------------------------------------------------------------------ Gemini
def test_gemini_grounded_report():
    if not configured(REAL.GEMINI_API_KEY):
        pytest.skip("GEMINI_API_KEY not configured")
    from backend.app.investigation_agent.llm_service import InvestigationAgentService, validate_report
    svc = InvestigationAgentService()
    svc.api_key, svc.model = REAL.GEMINI_API_KEY, REAL.LLM_MODEL
    pack = {"case": {"risk_score": 90, "risk_band": "HIGH", "required_action": "HOLD", "amount": 4800.0, "currency": "USD"},
            "evidence": [{"id": "E1", "category": "wallet_cluster", "text": "Possible layering indicator: 5-wallet loop within 4.5 hours (synthetic)."},
                         {"id": "E2", "category": "rule_factor", "text": "Amount is 40x the sender's average completed payment."},
                         {"id": "E3", "category": "rule_factor", "text": "Device not seen in the sender's completed payments."}],
            "categories": ["rule_factor", "wallet_cluster"]}
    report = asyncio.run(svc.generate_investigation_report(pack))
    if report["llm_status"] == "FALLBACK" and "unavailable" in (report["fallback_reason"] or ""):
        pytest.skip(f"Gemini temporarily unavailable: {report['fallback_reason']}")
    assert report["llm_status"] == "GEMINI", report["fallback_reason"]
    assert validate_report(report, ["E1", "E2", "E3"]) is None
    assert report["evidence_categories_sent"] == pack["categories"]


# ------------------------------------------------------------------ Etherscan
def test_etherscan_real_pyusd_transfers():
    spec = importlib.util.spec_from_file_location(
        "fetch_pyusd", Path(__file__).resolve().parent.parent / "scripts" / "fetch_pyusd_transfers.py")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    key = fetch._load_env_key()
    if not key:
        pytest.skip("ETHERSCAN_API_KEY not configured")
    result = fetch.fetch_pyusd_transfers(key, limit=5)
    assert len(result["transfers"]) == 5 and result["dropped"] == 0
    assert all(t["token_contract"] == fetch.PYUSD_CONTRACT and len(t["tx_hash"]) == 66 for t in result["transfers"])
