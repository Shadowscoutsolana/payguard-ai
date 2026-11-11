"""
Shared test setup.

1. Safety guard: tests must never call the real Gemini API (cost, quota, flakiness, and sending
   data to a third party). Any test that needs Gemini behaviour fakes the HTTP response instead.
2. Test-only API keys, so tests never depend on (or reveal) the real keys in .env.
"""

import pytest

from backend.app.config import settings
from backend.app.investigation_agent import task_manager as task_manager_module
from tests.keys import ANALYST_KEYS, MERCHANT_KEY


@pytest.fixture(autouse=True)
def no_real_gemini_calls(monkeypatch):
    monkeypatch.setattr(task_manager_module.task_manager.service, "api_key", None)


@pytest.fixture(autouse=True)
def test_api_keys(monkeypatch):
    monkeypatch.setattr(settings, "MERCHANT_API_KEY", MERCHANT_KEY)
    monkeypatch.setattr(settings, "ANALYST_API_KEYS", ",".join(f"{a}:{k}" for a, k in ANALYST_KEYS.items()))
