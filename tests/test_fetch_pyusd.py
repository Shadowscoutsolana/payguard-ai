"""
Tests for scripts/fetch_pyusd_transfers.py (no network: the HTTP session is faked).
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_pyusd", Path(__file__).resolve().parent.parent / "scripts" / "fetch_pyusd_transfers.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

A, B = "0x" + "11" * 20, "0x" + "22" * 20


def row(contract=fetch.PYUSD_CONTRACT, decimals="6", value="2500000"):
    return {"hash": "0x" + "ab" * 32, "logIndex": "7", "blockNumber": "21000000", "timeStamp": "1730000000",
            "from": A, "to": B, "value": value, "contractAddress": contract, "tokenDecimal": decimals}


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        data = self.responses.pop(0)

        class R:
            def json(self_inner):
                return data
        return R()


def test_parses_official_pyusd_rows():
    out = fetch.parse_transfers([row()])
    t = out["transfers"][0]
    assert t["amount_pyusd"] == 2.5 and t["tx_hash"] == "0x" + "ab" * 32
    assert t["transfer_id"].endswith(":7") and t["timestamp"].endswith("Z")
    assert t["token_contract"] == fetch.PYUSD_CONTRACT


@pytest.mark.parametrize("bad", [row(contract="0x" + "99" * 20), row(decimals="18")])
def test_drops_rows_that_are_not_official_pyusd(bad):
    assert fetch.parse_transfers([bad]) == {"transfers": [], "dropped": 1}


def test_uses_v2_mainnet_and_official_contract(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    session = FakeSession([{"status": "1", "result": [row()]}])
    fetch.fetch_pyusd_transfers("KEY", address=A, limit=5, session=session)
    url, params = session.calls[0]
    assert url == "https://api.etherscan.io/v2/api"
    assert params["chainid"] == 1 and params["contractaddress"] == fetch.PYUSD_CONTRACT and params["address"] == A


def test_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    session = FakeSession([{"status": "0", "message": "NOTOK", "result": "Max rate limit reached"},
                           {"status": "1", "result": [row()]}])
    assert len(fetch.fetch_pyusd_transfers("KEY", session=session)["transfers"]) == 1
    assert len(session.calls) == 2


def test_api_error_is_reported_not_swallowed(monkeypatch):
    session = FakeSession([{"status": "0", "message": "NOTOK", "result": "Invalid API Key"}])
    with pytest.raises(fetch.EtherscanError):
        fetch.fetch_pyusd_transfers("BAD", session=session)


@pytest.mark.parametrize("key,address", [("", None), ("KEY", "0x3Fa...88c1")])
def test_input_validation(key, address):
    with pytest.raises(fetch.EtherscanError):
        fetch.fetch_pyusd_transfers(key, address=address, session=FakeSession([]))
