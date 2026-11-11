"""Test-only API keys (never used outside the test suite)."""

MERCHANT_KEY = "test-merchant-key-0123456789abcdef"
ANALYST_KEYS = {
    "analyst.priya": "test-analyst-priya-0123456789abcd",
    "analyst.rahul": "test-analyst-rahul-0123456789abcd",
}
MERCHANT_HEADERS = {"X-API-Key": MERCHANT_KEY}


def analyst_headers(analyst_id: str = "analyst.priya") -> dict:
    return {"X-API-Key": ANALYST_KEYS[analyst_id]}
