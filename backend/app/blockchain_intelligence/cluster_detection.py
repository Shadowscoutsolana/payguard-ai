"""
Layer 3: Blockchain Intelligence - Suspicious wallet-cluster detection.

Indicators are COMPUTED from the transfer graph (labels in the data file decide nothing):
  CIRCULAR_TRANSFER_CYCLE  a directed cycle of <= CYCLE_MAX_WALLETS wallets through the recipient,
                           all hops completed within CYCLE_WINDOW_HOURS (layering indicator)
  RAPID_FAN_OUT            pass-through dispersion (mule indicator): the recipient sends to
                           >= FAN_OUT_MIN_RECIPIENTS distinct wallets within FAN_OUT_WINDOW_HOURS, AND
                           the amount sent out is >= FAN_OUT_MIN_PASS_THROUGH of what it received in the
                           same period from at most FAN_OUT_MAX_FUNDING_SOURCES senders.
                           (Recipient count alone also matches busy service wallets such as exchanges;
                           a real Ethereum wallet with 24 payouts but only 41% pass-through showed this.)

Every finding carries its evidence: the exact transfers, their time window and provenance.
These are investigative leads ("possible fraud-ring indicators"), NOT proof of fraud or of
common ownership; public addresses are pseudonymous.

Availability is reported separately from risk via "wallet_signal":
  NOT_PROVIDED (no wallet linked) / UNKNOWN (no intelligence) / KNOWN (wallet in our data)
"""

from datetime import timedelta
from typing import Any, Dict, List, Optional

from .graph import BlockchainGraphEngine, parse_ts
from .wallet_profile import WalletProfileService

CYCLE_MAX_WALLETS = 6
CYCLE_WINDOW_HOURS = 72
FAN_OUT_MIN_RECIPIENTS = 4
FAN_OUT_WINDOW_HOURS = 24
FAN_OUT_MIN_PASS_THROUGH = 0.8
FAN_OUT_MAX_FUNDING_SOURCES = 2

PENALTIES = {"CIRCULAR_TRANSFER_CYCLE": 50, "RAPID_FAN_OUT": 40}
ALLOWLISTED_MERCHANT_CREDIT = -10


def _evidence(transfers: List[Dict[str, Any]]) -> Dict[str, Any]:
    times = [parse_ts(t["timestamp"]) for t in transfers]
    return {
        "time_window_start": min(times).isoformat(),
        "time_window_end": max(times).isoformat(),
        "window_hours": round((max(times) - min(times)).total_seconds() / 3600, 2),
        "provenance": sorted({t.get("provenance", "unknown") for t in transfers}),
        "transfers": [{k: t.get(k) for k in ("transfer_id", "from", "to", "amount_pyusd", "timestamp",
                                             "tx_hash", "provenance")} for t in transfers],
    }


class ClusterDetector:
    def __init__(self, graph_engine: BlockchainGraphEngine, wallet_service: WalletProfileService):
        self.graph_engine = graph_engine
        self.wallet_service = wallet_service

    # ---------------------------------------------------------------- indicators
    def _cycle_indicator(self, address: str) -> Optional[Dict[str, Any]]:
        best = None
        for cycle in self.graph_engine.cycles_through(address, CYCLE_MAX_WALLETS):
            transfers = self.graph_engine.cycle_transfers(cycle)
            ev = _evidence(transfers)
            if ev["window_hours"] <= CYCLE_WINDOW_HOURS and (best is None or ev["window_hours"] < best["evidence"]["window_hours"]):
                best = {
                    "pattern_type": "CIRCULAR_TRANSFER_CYCLE",
                    "num_wallets": len(cycle),
                    "details": (f"Possible layering indicator: recipient wallet is part of a {len(cycle)}-wallet "
                                f"circular PYUSD transfer loop completed within {ev['window_hours']:g} hours."),
                    "evidence": ev,
                }
        return best

    def _fan_out_indicator(self, address: str) -> Optional[Dict[str, Any]]:
        outs = self.graph_engine.out_transfers(address)
        window = timedelta(hours=FAN_OUT_WINDOW_HOURS)
        for i, first in enumerate(outs):
            start = parse_ts(first["timestamp"])
            in_window = [t for t in outs[i:] if parse_ts(t["timestamp"]) - start <= window]
            recipients = {t["to"] for t in in_window}
            if len(recipients) < FAN_OUT_MIN_RECIPIENTS:
                continue
            # Funding: inflows from the window before the first payout up to the last payout.
            end = parse_ts(in_window[-1]["timestamp"])
            funding = [t for t in self.graph_engine.in_transfers(address)
                       if start - window <= parse_ts(t["timestamp"]) <= end]
            funded = sum(t["amount_pyusd"] for t in funding)
            dispersed = sum(t["amount_pyusd"] for t in in_window)
            pass_through = dispersed / funded if funded > 0 else 0.0
            sources = {t["from"] for t in funding}
            if pass_through < FAN_OUT_MIN_PASS_THROUGH or not (1 <= len(sources) <= FAN_OUT_MAX_FUNDING_SOURCES):
                continue  # many payouts but not a pass-through of a few deposits (e.g. a service wallet)
            ev = _evidence(funding + in_window)
            ev["pass_through_ratio"] = round(pass_through, 3)
            ev["funding_sources"] = len(sources)
            return {
                "pattern_type": "RAPID_FAN_OUT",
                "num_wallets": len(recipients) + 1,
                "details": (f"Possible mule-dispersion indicator: recipient wallet passed {pass_through:.0%} of "
                            f"{funded:,.2f} PYUSD received from {len(sources)} source(s) on to "
                            f"{len(recipients)} distinct wallets within {ev['window_hours']:g} hours."),
                "evidence": ev,
            }
        return None

    # ---------------------------------------------------------------- analysis
    def analyze_cluster(self, address: Optional[str]) -> Dict[str, Any]:
        if not address or not address.strip():
            return {"wallet_signal": "NOT_PROVIDED", "wallet_provenance": None, "has_cluster_flag": False,
                    "cluster_risk_penalty": 0, "findings": None, "indicators": [], "reasons": []}

        profile = self.wallet_service.get_wallet_profile(address)
        if profile.get("intelligence_status") != "KNOWN":
            return {"wallet_signal": "UNKNOWN", "wallet_provenance": "unavailable", "has_cluster_flag": False,
                    "cluster_risk_penalty": 0, "findings": None, "indicators": [], "reasons": []}

        addr = profile["address"]
        indicators = [i for i in (self._cycle_indicator(addr), self._fan_out_indicator(addr)) if i]
        for ind in indicators:
            ind["cluster_id"] = profile.get("cluster_id")

        if indicators:
            strongest = max(indicators, key=lambda i: PENALTIES[i["pattern_type"]])
            penalty = PENALTIES[strongest["pattern_type"]]
            reasons = [i["details"] for i in indicators]
        else:
            strongest = None
            penalty = ALLOWLISTED_MERCHANT_CREDIT if profile.get("allowlisted_merchant") else 0
            reasons = []

        return {
            "wallet_signal": "KNOWN",
            "wallet_provenance": profile.get("provenance"),
            "wallet_label": profile.get("label"),
            "allowlisted_merchant": profile.get("allowlisted_merchant", False),
            "has_cluster_flag": bool(indicators),
            "cluster_risk_penalty": penalty,
            "findings": strongest,          # strongest indicator, with evidence
            "indicators": indicators,       # all indicators found
            "reasons": reasons,
        }

    def cluster_report(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        """Everything we hold about one cluster: members, transfers and computed indicators."""
        meta = self.wallet_service.clusters.get(cluster_id)
        if not meta:
            return None
        members = meta["members"]
        transfers = [t for t in self.graph_engine.data["transfers"] if t.get("cluster_id") == cluster_id]
        seen, indicators = set(), []
        for m in members:
            for ind in self.analyze_cluster(m)["indicators"]:
                sig = (ind["pattern_type"], tuple(sorted(x["transfer_id"] or "" for x in ind["evidence"]["transfers"])))
                if sig not in seen:
                    seen.add(sig)
                    indicators.append({**ind, "detected_at_wallet": m})
        return {
            "cluster_id": cluster_id,
            "description": meta["description"],
            "provenance": meta["provenance"],
            "target_wallet": meta["target_wallet"],
            "members": [self.wallet_service.get_wallet_profile(m) for m in members],
            "transfers": sorted(transfers, key=lambda t: t["timestamp"]),
            "indicators": indicators,
            "disclaimer": ("Possible fraud-ring indicators for investigation, not proof of fraud or of common "
                           "ownership. Public addresses are pseudonymous."),
        }
