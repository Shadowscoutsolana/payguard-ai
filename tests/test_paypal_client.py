"""
Tests for the PayPal Orders v2 client (Task 1.2: CAPTURE intent only).
Real-mode requests are intercepted by a fake HTTP client; nothing is sent to PayPal.
"""

import asyncio
import pytest

from backend.app.paypal_integration import client as client_module
from backend.app.paypal_integration.client import PayPalClient, ORDER_INTENT


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class FakeAsyncClient:
    """Records every POST so tests can inspect URL, headers and body."""
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None, **kwargs):
        FakeAsyncClient.calls.append({"url": url, "json": json, "headers": headers or {}})
        if url.endswith("/capture"):
            return FakeResponse({"id": "ORDER-REAL-1", "status": "COMPLETED"})
        return FakeResponse({"id": "ORDER-REAL-1", "status": "CREATED", "links": []})


@pytest.fixture
def real_mode_client(monkeypatch):
    FakeAsyncClient.calls = []
    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    c = PayPalClient()
    c.is_configured = True

    async def fake_token():
        return "test-token"

    monkeypatch.setattr(c, "get_access_token", fake_token)
    return c


def test_intent_is_capture():
    assert ORDER_INTENT == "CAPTURE"


def test_mock_order_uses_capture_intent():
    c = PayPalClient()
    c.is_configured = False
    order = asyncio.run(c.create_order(amount=10.0))
    assert order["intent"] == "CAPTURE"
    assert order["mock_mode"] is True


def test_real_create_order_sends_capture_intent(real_mode_client):
    asyncio.run(real_mode_client.create_order(amount=45.5, currency="USD"))
    call = FakeAsyncClient.calls[-1]
    assert call["url"].endswith("/v2/checkout/orders")
    assert call["json"]["intent"] == "CAPTURE"
    assert call["json"]["purchase_units"][0]["amount"] == {"currency_code": "USD", "value": "45.50"}
    assert call["headers"]["PayPal-Request-Id"].startswith("payguard-create-")


def test_real_capture_uses_order_capture_endpoint_idempotently(real_mode_client):
    asyncio.run(real_mode_client.capture_order("ORDER-REAL-1"))
    asyncio.run(real_mode_client.capture_order("ORDER-REAL-1"))
    first, second = FakeAsyncClient.calls[-2:]
    assert first["url"].endswith("/v2/checkout/orders/ORDER-REAL-1/capture")
    assert "/authorizations/" not in first["url"]
    # Same request id on retry -> PayPal will not capture twice.
    assert first["headers"]["PayPal-Request-Id"] == second["headers"]["PayPal-Request-Id"]


def test_missing_token_raises_instead_of_sending_bearer_none(monkeypatch):
    c = PayPalClient()
    c.is_configured = True

    async def no_token():
        return None

    monkeypatch.setattr(c, "get_access_token", no_token)
    with pytest.raises(RuntimeError):
        asyncio.run(c.create_order(amount=10.0))
