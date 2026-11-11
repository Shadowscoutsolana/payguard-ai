"""
Layer 1: Transaction Intelligence Feature Extractor.
Validates/normalizes payment fields and computes amount deviation, velocity and corridor risk.

Historical values (average amount, 24 h velocity) come from the server-computed behavioural
profile (see behavioral_detection.anomaly_detector), never from the client request.
"""

from typing import Any, Dict, Optional


class TransactionFeatureExtractor:
    def __init__(self):
        # High risk international payment corridors for demonstration
        self.high_risk_corridors = {
            ("US", "KY"),  # Cayman Islands
            ("US", "VG"),  # British Virgin Islands
            ("US", "PA"),  # Panama
            ("US", "SC"),  # Seychelles
        }

    def extract_features(self, payload: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        `profile` is the sender baseline built server-side. Without a baseline the amount
        cannot be compared, so the deviation ratio is neutral (1.0) and the risk engine's
        "unknown is never LOW" policy applies instead.
        """
        profile = profile or {}
        amount = float(payload.get("amount", 0.0))
        sender_country = payload.get("sender_country", "US")
        recipient_country = payload.get("recipient_country", "US")

        hist_avg = profile.get("avg_amount")
        baseline_known = hist_avg is not None and hist_avg > 0
        effective_avg = float(hist_avg) if baseline_known else max(amount, 1.0)
        velocity_24h = int(profile.get("attempts_last_24h", 0))

        # 1. Amount deviation ratio (current amount / sender's historical average)
        amount_ratio = amount / max(effective_avg, 1.0)
        is_large_amount = 1.0 if amount >= 2500.0 else 0.0

        # 2. Corridor cross-border indicator
        is_cross_border = 1.0 if sender_country != recipient_country else 0.0
        is_high_risk_corridor = (
            1.0 if (sender_country, recipient_country) in self.high_risk_corridors else 0.0
        )

        # 3. Velocity (attempts by this sender in the last 24 h, before this one)
        is_high_velocity = 1.0 if velocity_24h > 5 else 0.0

        return {
            "amount": amount,
            "historical_avg_amount": effective_avg,
            "baseline_known": baseline_known,
            "amount_deviation_ratio": round(amount_ratio, 2),
            "is_large_amount": is_large_amount,
            "is_cross_border": is_cross_border,
            "is_high_risk_corridor": is_high_risk_corridor,
            "velocity_last_24h": velocity_24h,
            "is_high_velocity": is_high_velocity,
            # Completed payments this recipient received from any sender (ML model feature).
            "recipient_prior_receipts": int(profile.get("recipient_prior_receipts", 0)),
        }
