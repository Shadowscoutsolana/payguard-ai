"""
Layer 3: Blockchain Intelligence - Wallet Profiles.

Looks up what we know about a wallet from the cached data (synthetic demo wallets and, if
fetched, real public-chain PYUSD transfers). EXACT address match only (case-insensitive):
a fragment such as "0x" never resolves to a known wallet, and an address with no data is
UNKNOWN, never "clean".
"""

from typing import Any, Dict, Optional

from .graph import load_blockchain_data


class WalletProfileService:
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or load_blockchain_data()
        self.wallets = self.data["wallets"]
        self.clusters = self.data["clusters"]

    def get_wallet_profile(self, address: str) -> Optional[Dict[str, Any]]:
        if not address:
            return None
        key = address.strip().lower()
        w = self.wallets.get(key)
        if w is None:
            return {
                "intelligence_status": "UNKNOWN",
                "address": address,
                "label": "No wallet intelligence available",
                "provenance": "unavailable",
                "is_flagged": None,  # unknown, not False
                "cluster_id": None,
                "allowlisted_merchant": False,
                "description": "No cached or fetched PYUSD activity for this address; treated as UNKNOWN, not clean.",
            }
        cluster = self.clusters.get(w.get("cluster_id")) or {}
        return {
            "intelligence_status": "KNOWN",
            "address": key,
            "label": w.get("label"),
            "provenance": w.get("provenance", "synthetic"),
            "cluster_id": w.get("cluster_id"),
            "allowlisted_merchant": bool(w.get("allowlisted_merchant")),
            "description": cluster.get("description", ""),
        }
