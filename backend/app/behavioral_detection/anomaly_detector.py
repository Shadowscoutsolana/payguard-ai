"""
Layer 2: Behavioural Detection.

Builds a sender's behavioural baseline on the SERVER from:
  1. completed (captured) payments stored in payguard.db, and
  2. a clearly labelled synthetic seed history for the demo personas
     (data/demo_behaviour_history.json).
Client-supplied history values are never used.

Only COMPLETED payments shape the baseline (average amount, known devices, known
recipients), so blocked or rejected attempts cannot "train" a fraudster's own profile.
Velocity counts ALL attempts in the last 24 h, because bursts of attempts are the signal.

device_id values are simulated labels in this prototype, not real device fingerprints.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SEED_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "demo_behaviour_history.json"


class BehavioralAnomalyDetector:
    def __init__(self, seed_path: Path = SEED_PATH):
        self.seed_history: Dict[str, list] = {}
        if seed_path.exists():
            with open(seed_path, "r", encoding="utf-8") as f:
                self.seed_history = json.load(f).get("senders", {})

    def seed_receipts_for(self, recipient_id: Optional[str]) -> int:
        """How many seed payments (any sender) went to this recipient."""
        if not recipient_id:
            return 0
        return sum(1 for payments in self.seed_history.values() for p in payments
                   if p.get("recipient_id") == recipient_id)

    def build_profile(self, sender_id: str,
                      completed_payments: Iterable[Dict[str, Any]] = (),
                      attempts_last_24h: int = 0,
                      recipient_id: Optional[str] = None,
                      stored_recipient_receipts: int = 0) -> Dict[str, Any]:
        """
        Baseline for one sender. `completed_payments` are captured payments from the database
        (dicts with amount, recipient_id, device_id); the synthetic seed is added for demo personas.
        recipient_prior_receipts = completed payments this recipient received from ANY sender
        (the ML model's recipient feature).
        """
        seed = [dict(p, source="synthetic_seed") for p in self.seed_history.get(sender_id, [])]
        stored = [dict(p, source="stored_transactions") for p in completed_payments]
        history = seed + stored

        amounts = [float(p["amount"]) for p in history if p.get("amount") is not None]
        return {
            "sender_id": sender_id,
            "baseline_status": "KNOWN" if history else "UNKNOWN",
            "history_count": len(history),
            "seed_count": len(seed),
            "stored_count": len(stored),
            "avg_amount": round(sum(amounts) / len(amounts), 2) if amounts else None,
            "known_devices": sorted({p["device_id"] for p in history if p.get("device_id")}),
            "known_recipients": sorted({p["recipient_id"] for p in history if p.get("recipient_id")}),
            "attempts_last_24h": int(attempts_last_24h),
            "recipient_prior_receipts": self.seed_receipts_for(recipient_id) + int(stored_recipient_receipts),
        }

    def detect_anomalies(self, payload: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compares this payment with the sender's baseline. Without a profile (or with an
        UNKNOWN baseline) nothing can be confirmed as familiar, so device and recipient
        are treated as unseen and baseline_status is UNKNOWN.
        """
        profile = profile or self.build_profile(payload.get("sender_id", ""))
        device = payload.get("device_id")
        recipient = payload.get("recipient_id")

        is_unknown_device = 0.0 if device and device in profile["known_devices"] else 1.0
        is_new_recipient = 0.0 if recipient and recipient in profile["known_recipients"] else 1.0

        behavior_penalty = (25 if is_unknown_device else 0) + (15 if is_new_recipient else 0)
        return {
            "is_unknown_device": is_unknown_device,
            "device_provided": bool(device),
            "is_new_recipient": is_new_recipient,
            "behavior_risk_penalty": behavior_penalty,
            "device_anomaly_flag": bool(is_unknown_device),
            "baseline_status": profile["baseline_status"],
        }
