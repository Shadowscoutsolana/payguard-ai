r"""
Fetch REAL PYUSD (PayPal USD) ERC-20 transfers from Ethereum mainnet via the Etherscan API V2.

- Network: Ethereum mainnet (chainid=1).
- Token: official PYUSD contract 0x6c3ea9036406852006290770BEdFcAbA0e23A0e8, as published by
  Paxos (https://docs.paxos.com/stablecoin/pyusd/pyusd-faq) and PayPal (help article HELP1005).
- API: https://api.etherscan.io/v2/api (V1 was retired on 15 Aug 2025). Needs a free Etherscan API key.
- Every returned row is re-checked: contract address must equal the official PYUSD contract and
  tokenDecimal must be 6; anything else is dropped and reported.
- Output: data/real_pyusd_transfers.json, which the graph engine loads with provenance
  "real_public_chain" (real tx hashes). Public addresses are pseudonymous; no labels are invented.

Usage (PowerShell, from the project folder; set the key in .env as ETHERSCAN_API_KEY=...):
    .\.venv\Scripts\python.exe scripts\fetch_pyusd_transfers.py --address 0x<wallet> --limit 100
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

PYUSD_CONTRACT = "0x6c3ea9036406852006290770bedfcaba0e23a0e8"
PYUSD_DECIMALS = 6
ETHEREUM_MAINNET_CHAIN_ID = 1
ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "real_pyusd_transfers.json"
ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


class EtherscanError(RuntimeError):
    pass


def _get_with_retries(params: Dict[str, Any], retries: int = 4, session=requests) -> Dict[str, Any]:
    """GET with exponential backoff on rate limits / transient errors. Never logs the API key."""
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(ETHERSCAN_V2_URL, params=params, timeout=20)
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                raise EtherscanError(f"Network/parse error after {retries} attempts: {type(exc).__name__}")
            time.sleep(delay); delay *= 2
            continue
        message = f"{data.get('message', '')} {data.get('result', '')}".lower()
        if data.get("status") == "1":
            return data
        if "no transactions found" in message:
            return {"status": "1", "result": []}
        if "rate limit" in message and attempt < retries:
            time.sleep(delay); delay *= 2
            continue
        raise EtherscanError(f"Etherscan returned an error: {data.get('message')} ({str(data.get('result'))[:120]})")
    raise EtherscanError("Unreachable")


def parse_transfers(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keeps only official-PYUSD rows; returns {"transfers": [...], "dropped": n}."""
    kept, dropped = [], 0
    for r in rows:
        if (r.get("contractAddress", "").lower() != PYUSD_CONTRACT
                or int(r.get("tokenDecimal", -1)) != PYUSD_DECIMALS
                or not ADDRESS_RE.match(r.get("from", "")) or not ADDRESS_RE.match(r.get("to", ""))):
            dropped += 1
            continue
        ts = datetime.fromtimestamp(int(r["timeStamp"]), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        kept.append({
            "transfer_id": f"{r['hash']}:{r.get('logIndex', '0')}",
            "from": r["from"].lower(),
            "to": r["to"].lower(),
            "amount_pyusd": round(int(r["value"]) / 10 ** PYUSD_DECIMALS, 6),
            "timestamp": ts,
            "tx_hash": r["hash"],
            "block_number": int(r["blockNumber"]),
            "token": "PYUSD",
            "token_contract": PYUSD_CONTRACT,
        })
    return {"transfers": kept, "dropped": dropped}


def fetch_pyusd_transfers(api_key: str, address: Optional[str] = None, limit: int = 100, session=requests) -> Dict[str, Any]:
    if not api_key:
        raise EtherscanError("No Etherscan API key. Set ETHERSCAN_API_KEY in .env (free at https://etherscan.io/apis).")
    if address and not ADDRESS_RE.match(address):
        raise EtherscanError("--address must be a full 42-character 0x address.")
    params = {"chainid": ETHEREUM_MAINNET_CHAIN_ID, "module": "account", "action": "tokentx",
              "contractaddress": PYUSD_CONTRACT, "page": 1, "offset": max(1, min(limit, 1000)),
              "sort": "desc", "apikey": api_key}
    if address:
        params["address"] = address
    data = _get_with_retries(params, session=session)
    return parse_transfers(data.get("result") or [])


def _load_env_key() -> str:
    key = os.getenv("ETHERSCAN_API_KEY", "")
    env = Path(__file__).resolve().parent.parent / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ETHERSCAN_API_KEY="):
                key = line.split("=", 1)[1].strip()
    return "" if key.startswith("your_") else key


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real PYUSD transfers (Ethereum mainnet, Etherscan V2)")
    parser.add_argument("--address", default="", help="Optional wallet to fetch transfers for (full 0x address)")
    parser.add_argument("--limit", type=int, default=100, help="Max transfers (1-1000)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    try:
        result = fetch_pyusd_transfers(_load_env_key(), address=args.address or None, limit=args.limit)
    except EtherscanError as exc:
        print(f"ERROR: {exc}")
        return 1

    payload = {
        "_metadata": {
            "provenance": "real_public_chain",
            "source": "Etherscan API V2 (account/tokentx)",
            "network": "Ethereum mainnet (chainid=1)",
            "token_contract": PYUSD_CONTRACT,
            "address_filter": args.address or None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "transfer_count": len(result["transfers"]),
            "rows_dropped_failed_contract_check": result["dropped"],
            "note": "Public addresses are pseudonymous. No labels or risk judgements are attached by this script.",
        },
        "transfers": result["transfers"],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(result['transfers'])} real PYUSD transfers to {args.output} "
          f"({result['dropped']} rows dropped by the contract check).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
