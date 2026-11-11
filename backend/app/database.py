"""
PayGuard AI - SQLite Storage Layer.
Thread-safe database manager providing persistent storage for:
- Transactions & pre-capture scoring evaluations
- Analyst alerts & human-in-the-loop review decisions
- PayPal payment lifecycle events
- Compliance & security audit logs
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

DB_PATH = Path(os.getenv("PAYGUARD_DB_PATH")
               or Path(__file__).resolve().parent.parent.parent / "payguard.db")


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates the required tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Transactions & Pre-Capture Evaluations
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                sender_id TEXT,
                recipient_id TEXT,
                amount REAL,
                currency TEXT,
                recipient_wallet TEXT,
                device_id TEXT,
                risk_score INTEGER,
                risk_band TEXT,
                action TEXT,
                decision_latency_ms REAL,
                reasons TEXT,
                contributing_factors TEXT,
                raw_payload TEXT,
                signal_availability TEXT,
                policy_overrides TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            tx_cols = {r[1] for r in cursor.execute("PRAGMA table_info(transactions)")}
            for column in ("signal_availability TEXT", "policy_overrides TEXT",
                           "model_explanations TEXT", "score_breakdown TEXT"):
                if column.split()[0] not in tx_cols:
                    cursor.execute(f"ALTER TABLE transactions ADD COLUMN {column}")

            # 2. Analyst Alerts Queue & Human Decisions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                transaction_id TEXT,
                status TEXT DEFAULT 'PENDING_REVIEW',
                risk_score INTEGER,
                risk_band TEXT,
                amount REAL,
                currency TEXT,
                recipient_id TEXT,
                recipient_wallet TEXT,
                cluster_findings TEXT,
                reasons TEXT,
                llm_report TEXT,
                analyst_decision TEXT,
                analyst_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
            )
            """)

            # 3. PayPal Payment Lifecycle Events & Webhooks
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS payment_events (
                event_id TEXT PRIMARY KEY,
                order_id TEXT,
                event_type TEXT,
                status TEXT,
                details TEXT,
                verification_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 4. PayPal Orders <-> Risk Decisions (server-side source of truth for the capture gate)
            # status lifecycle:
            #   CREATED -> APPROVED (buyer, verified with PayPal) -> RISK_EVALUATED
            #   -> CAPTURE_IN_PROGRESS -> CAPTURED | CAPTURE_FAILED
            #   CREATED -> CANCELLED (buyer cancelled on PayPal)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS payment_orders (
                order_id TEXT PRIMARY KEY,
                transaction_id TEXT UNIQUE,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'CREATED',
                paypal_status TEXT,
                mock_mode INTEGER NOT NULL DEFAULT 0,
                risk_score INTEGER,
                risk_band TEXT,
                risk_action TEXT,
                step_up_status TEXT,
                analyst_decision TEXT,
                capture_id TEXT,
                last_error TEXT,
                payer_id TEXT,
                approval_source TEXT,
                approved_at TIMESTAMP,
                approval_url TEXT,
                checkout_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            # Lightweight migration for databases created before these columns existed.
            existing = {r[1] for r in cursor.execute("PRAGMA table_info(payment_orders)")}
            for column in ("payer_id TEXT", "approval_source TEXT", "approved_at TIMESTAMP",
                           "approval_url TEXT", "checkout_context TEXT",
                           "analyst_id TEXT", "analyst_notes TEXT", "decided_at TIMESTAMP"):
                if column.split()[0] not in existing:
                    cursor.execute(f"ALTER TABLE payment_orders ADD COLUMN {column}")

            alert_cols = {r[1] for r in cursor.execute("PRAGMA table_info(alerts)")}
            for column in ("analyst_id TEXT", "decided_at TIMESTAMP", "capture_status TEXT", "capture_detail TEXT"):
                if column.split()[0] not in alert_cols:
                    cursor.execute(f"ALTER TABLE alerts ADD COLUMN {column}")

            # 5. Step-up verification challenges (simulated OTP). Only a keyed hash of the code is stored.
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS step_up_challenges (
                challenge_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempts INTEGER NOT NULL DEFAULT 0,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """)

            # 6. Immutable Compliance Audit Logs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT,
                entity_id TEXT,
                action TEXT,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.commit()

    # --- Transactions API ---
    def save_transaction(self, record: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO transactions (
                transaction_id, sender_id, recipient_id, amount, currency,
                recipient_wallet, device_id, risk_score, risk_band, action,
                decision_latency_ms, reasons, contributing_factors, raw_payload,
                signal_availability, policy_overrides, model_explanations, score_breakdown, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("transaction_id"),
                record.get("sender_id"),
                record.get("recipient_id"),
                record.get("amount"),
                record.get("currency", "USD"),
                record.get("recipient_wallet"),
                record.get("device_id"),
                record.get("risk_score"),
                record.get("risk_band"),
                record.get("action"),
                record.get("decision_latency_ms"),
                json.dumps(record.get("reasons", [])),
                json.dumps(record.get("contributing_factors", [])),
                json.dumps(record.get("raw_payload", {})),
                json.dumps(record.get("signal_availability", {})),
                json.dumps(record.get("policy_overrides", [])),
                json.dumps(record.get("model_explanations", [])),
                json.dumps(record.get("score_breakdown", {})),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
        self.log_audit("TRANSACTION", record.get("transaction_id"), "EVALUATE_PRE_CAPTURE", record)

    def get_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["reasons"] = json.loads(data["reasons"]) if data["reasons"] else []
            data["contributing_factors"] = json.loads(data["contributing_factors"]) if data["contributing_factors"] else []
            data["raw_payload"] = json.loads(data["raw_payload"]) if data["raw_payload"] else {}
            data["signal_availability"] = json.loads(data["signal_availability"]) if data.get("signal_availability") else {}
            data["policy_overrides"] = json.loads(data["policy_overrides"]) if data.get("policy_overrides") else []
            data["model_explanations"] = json.loads(data["model_explanations"]) if data.get("model_explanations") else []
            data["score_breakdown"] = json.loads(data["score_breakdown"]) if data.get("score_breakdown") else {}
            return data

    # --- Behavioural history (server-side baseline) ---
    def get_completed_payments(self, sender_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Sender's payments that were actually CAPTURED (blocked/rejected attempts excluded)."""
        with self._get_connection() as conn:
            rows = conn.execute("""
            SELECT t.amount, t.recipient_id, t.device_id, t.created_at
            FROM transactions t JOIN payment_orders o ON o.transaction_id = t.transaction_id
            WHERE t.sender_id = ? AND o.status = 'CAPTURED'
            ORDER BY t.created_at DESC LIMIT ?
            """, (sender_id, limit)).fetchall()
            return [dict(r) for r in rows]

    def count_completed_receipts(self, recipient_id: str) -> int:
        """Captured payments received by this recipient from any sender."""
        with self._get_connection() as conn:
            return conn.execute("""
            SELECT COUNT(*) FROM transactions t JOIN payment_orders o ON o.transaction_id = t.transaction_id
            WHERE t.recipient_id = ? AND o.status = 'CAPTURED'
            """, (recipient_id,)).fetchone()[0]

    def count_attempts_since(self, sender_id: str, since_iso: str) -> int:
        """All risk evaluations by this sender since `since_iso` (any outcome)."""
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM transactions WHERE sender_id = ? AND created_at >= ?",
                                (sender_id, since_iso)).fetchone()[0]

    def list_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["reasons"] = json.loads(d["reasons"]) if d["reasons"] else []
                results.append(d)
            return results

    # --- Alerts API ---
    def save_alert(self, alert_data: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO alerts (
                alert_id, transaction_id, status, risk_score, risk_band,
                amount, currency, recipient_id, recipient_wallet,
                cluster_findings, reasons, llm_report, analyst_decision, analyst_notes,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_data.get("alert_id"),
                alert_data.get("transaction_id"),
                alert_data.get("status", "PENDING_REVIEW"),
                alert_data.get("risk_score"),
                alert_data.get("risk_band"),
                alert_data.get("amount"),
                alert_data.get("currency", "USD"),
                alert_data.get("recipient_id"),
                alert_data.get("recipient_wallet"),
                json.dumps(alert_data.get("cluster_findings")) if alert_data.get("cluster_findings") else None,
                json.dumps(alert_data.get("reasons", [])),
                json.dumps(alert_data.get("llm_report")) if alert_data.get("llm_report") else None,
                alert_data.get("analyst_decision"),
                alert_data.get("analyst_notes"),
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

    def update_alert_llm_report(self, alert_id: str, llm_report: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE alerts
            SET llm_report = ?, updated_at = ?
            WHERE alert_id = ?
            """, (json.dumps(llm_report), datetime.now(timezone.utc).isoformat(), alert_id))
            conn.commit()

    def resolve_alert(self, alert_id: str, decision: str, notes: Optional[str],
                      analyst_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Records the analyst decision exactly once (only while PENDING_REVIEW).
        Returns the updated alert, or None if it was not pending (missing or already decided).
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE alerts
            SET status = 'RESOLVED', analyst_decision = ?, analyst_notes = ?, analyst_id = ?,
                decided_at = ?, updated_at = ?
            WHERE alert_id = ? AND status = 'PENDING_REVIEW'
            """, (decision, notes, analyst_id, now, now, alert_id))
            conn.commit()
            if cur.rowcount != 1:
                return None

        self.log_audit("ALERT", alert_id, f"ANALYST_{decision}",
                       {"analyst_id": analyst_id, "notes": notes, "decided_at": now})
        return self.get_alert(alert_id)

    def set_alert_capture_status(self, alert_id: str, capture_status: str, detail: str) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE alerts SET capture_status = ?, capture_detail = ?, updated_at = ? WHERE alert_id = ?",
                         (capture_status, detail, datetime.now(timezone.utc).isoformat(), alert_id))
            conn.commit()
        self.log_audit("ALERT", alert_id, f"CAPTURE_{capture_status}", {"detail": detail})

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._parse_alert_row(row)

    def list_alerts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM alerts WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cursor.execute("SELECT * FROM alerts ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._parse_alert_row(r) for r in rows]

    def _parse_alert_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["cluster_findings"] = json.loads(data["cluster_findings"]) if data.get("cluster_findings") else None
        data["reasons"] = json.loads(data["reasons"]) if data.get("reasons") else []
        data["llm_report"] = json.loads(data["llm_report"]) if data.get("llm_report") else None
        return data

    # --- Payment Events & Webhooks API ---
    def log_payment_event(self, event_id: str, order_id: str, event_type: str, status: str, details: Any, verification_status: str = "SUCCESS"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO payment_events (
                event_id, order_id, event_type, status, details, verification_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                order_id,
                event_type,
                status,
                json.dumps(details) if isinstance(details, (dict, list)) else str(details),
                verification_status,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
        self.log_audit("PAYMENT_EVENT", event_id, event_type, details)

    def claim_webhook_event(self, event_id: str, order_id: str, event_type: str, details: Any) -> bool:
        """
        Idempotency for PayPal webhooks: stores a verified event exactly once.
        Returns False if this event id was already received (PayPal retries deliveries),
        in which case the caller must not process it again.
        """
        with self._get_connection() as conn:
            cur = conn.execute("""
            INSERT OR IGNORE INTO payment_events (
                event_id, order_id, event_type, status, details, verification_status, created_at
            ) VALUES (?, ?, ?, 'RECEIVED', ?, 'SUCCESS', ?)
            """, (event_id, order_id, event_type, json.dumps(details),
                  datetime.now(timezone.utc).isoformat()))
            conn.commit()
            claimed = cur.rowcount == 1
        self.log_audit("WEBHOOK", event_id, "WEBHOOK_RECEIVED" if claimed else "WEBHOOK_DUPLICATE_IGNORED",
                       {"event_type": event_type, "order_id": order_id})
        return claimed

    def set_webhook_event_result(self, event_id: str, status: str, note: str) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE payment_events SET status = ? WHERE event_id = ?", (status, event_id))
            conn.commit()
        self.log_audit("WEBHOOK", event_id, f"WEBHOOK_{status}", {"note": note})

    def update_order_paypal_status(self, order_id: str, paypal_status: str, reason: str) -> bool:
        """Records PayPal-side lifecycle changes (denied/refunded/reversed) reported by webhooks."""
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders SET paypal_status = ?, updated_at = ? WHERE order_id = ?
            """, (paypal_status, datetime.now(timezone.utc).isoformat(), order_id))
            conn.commit()
            ok = cur.rowcount == 1
        if ok:
            self.log_audit("PAYMENT_ORDER", order_id, f"PAYPAL_STATUS_{paypal_status}", {"reason": reason})
        return ok

    def list_payment_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM payment_events ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    # --- Payment Orders API (risk-gated capture state) ---
    def create_payment_order(self, order_id: str, amount: float, currency: str,
                              paypal_status: str, mock_mode: bool,
                              approval_url: Optional[str] = None,
                              checkout_context: Optional[Dict[str, Any]] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO payment_orders (order_id, amount, currency, status, paypal_status,
                                        mock_mode, approval_url, checkout_context, created_at, updated_at)
            VALUES (?, ?, ?, 'CREATED', ?, ?, ?, ?, ?, ?)
            """, (order_id, amount, currency, paypal_status, int(mock_mode), approval_url,
                  json.dumps(checkout_context or {}), now, now))
            conn.commit()
        self.log_audit("PAYMENT_ORDER", order_id, "ORDER_CREATED",
                       {"amount": amount, "currency": currency, "mock_mode": mock_mode})

    def get_payment_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM payment_orders WHERE order_id = ?", (order_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            data["checkout_context"] = json.loads(data["checkout_context"]) if data.get("checkout_context") else {}
            return data

    def get_payment_order_by_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT order_id FROM payment_orders WHERE transaction_id = ?",
                               (transaction_id,)).fetchone()
        return self.get_payment_order(row[0]) if row else None

    def record_order_analyst_decision(self, order_id: str, decision: str, analyst_id: str, notes: str) -> bool:
        """
        Stores the analyst decision on the order exactly once. Only orders the risk gate
        sent to a human are eligible: HOLD (high risk) or STEP_UP that FAILED.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders
            SET analyst_decision = ?, analyst_id = ?, analyst_notes = ?, decided_at = ?, updated_at = ?
            WHERE order_id = ? AND analyst_decision IS NULL
              AND status IN ('RISK_EVALUATED', 'CAPTURE_FAILED')
              AND (risk_action = 'HOLD' OR (risk_action = 'STEP_UP' AND step_up_status = 'FAILED'))
            """, (decision, analyst_id, notes, now, now, order_id))
            conn.commit()
            ok = cur.rowcount == 1
        if ok:
            self.log_audit("PAYMENT_ORDER", order_id, f"ANALYST_{decision}",
                           {"analyst_id": analyst_id, "notes": notes, "decided_at": now})
        return ok

    def mark_order_approved(self, order_id: str, payer_id: Optional[str], approval_source: str) -> bool:
        """
        Records buyer approval. Call ONLY after the order status was confirmed with PayPal's API
        (approval_source='PAYPAL_API') or in mock mode (approval_source='SIMULATED').
        Atomic: only a CREATED order can become APPROVED.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders
            SET status = 'APPROVED', paypal_status = 'APPROVED', payer_id = ?,
                approval_source = ?, approved_at = ?, updated_at = ?
            WHERE order_id = ? AND status = 'CREATED'
            """, (payer_id, approval_source, now, now, order_id))
            conn.commit()
            ok = cur.rowcount == 1
        if ok:
            self.log_audit("PAYMENT_ORDER", order_id, "BUYER_APPROVAL_VERIFIED",
                           {"payer_id": payer_id, "approval_source": approval_source})
        return ok

    def mark_order_cancelled(self, order_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders SET status = 'CANCELLED', updated_at = ?
            WHERE order_id = ? AND status = 'CREATED'
            """, (datetime.now(timezone.utc).isoformat(), order_id))
            conn.commit()
            ok = cur.rowcount == 1
        if ok:
            self.log_audit("PAYMENT_ORDER", order_id, "BUYER_CANCELLED", {})
        return ok

    def attach_risk_decision(self, order_id: str, transaction_id: str, risk_score: int,
                             risk_band: str, risk_action: str) -> bool:
        """
        Links a risk evaluation to an order exactly once.
        Atomic: only succeeds while the order is APPROVED (by the buyer) and has no decision yet,
        so a caller cannot re-score an order to replace a HIGH decision with a LOW one.
        """
        step_up_status = "PENDING" if risk_action == "STEP_UP" else None
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders
            SET transaction_id = ?, risk_score = ?, risk_band = ?, risk_action = ?,
                step_up_status = ?, status = 'RISK_EVALUATED', updated_at = ?
            WHERE order_id = ? AND status = 'APPROVED' AND transaction_id IS NULL
            """, (transaction_id, risk_score, risk_band, risk_action, step_up_status,
                  datetime.now(timezone.utc).isoformat(), order_id))
            conn.commit()
            ok = cur.rowcount == 1
        if ok:
            self.log_audit("PAYMENT_ORDER", order_id, "RISK_DECISION_ATTACHED", {
                "transaction_id": transaction_id, "risk_score": risk_score,
                "risk_band": risk_band, "risk_action": risk_action})
        return ok

    def claim_order_for_capture(self, order_id: str) -> bool:
        """
        Atomically moves an order into CAPTURE_IN_PROGRESS.
        Only one concurrent request can win, which blocks duplicate captures.
        """
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders SET status = 'CAPTURE_IN_PROGRESS', updated_at = ?
            WHERE order_id = ? AND status IN ('RISK_EVALUATED', 'CAPTURE_FAILED')
            """, (datetime.now(timezone.utc).isoformat(), order_id))
            conn.commit()
            return cur.rowcount == 1

    def finish_capture(self, order_id: str, success: bool, capture_id: Optional[str],
                       paypal_status: Optional[str], error: Optional[str]) -> None:
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE payment_orders
            SET status = ?, capture_id = ?, paypal_status = ?, last_error = ?, updated_at = ?
            WHERE order_id = ? AND status = 'CAPTURE_IN_PROGRESS'
            """, ("CAPTURED" if success else "CAPTURE_FAILED", capture_id, paypal_status,
                  error, datetime.now(timezone.utc).isoformat(), order_id))
            conn.commit()
        self.log_audit("PAYMENT_ORDER", order_id,
                       "CAPTURE_SUCCEEDED" if success else "CAPTURE_FAILED",
                       {"capture_id": capture_id, "paypal_status": paypal_status, "error": error})

    # --- Step-up verification (simulated OTP) ---
    def create_step_up_challenge(self, challenge_id: str, order_id: str, code_hash: str, expires_at: str) -> None:
        """Supersedes any still-pending challenge for the order, then stores the new one."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("UPDATE step_up_challenges SET status = 'SUPERSEDED' "
                         "WHERE order_id = ? AND status = 'PENDING'", (order_id,))
            conn.execute("""
            INSERT INTO step_up_challenges (challenge_id, order_id, code_hash, status, attempts, expires_at, created_at)
            VALUES (?, ?, ?, 'PENDING', 0, ?, ?)
            """, (challenge_id, order_id, code_hash, expires_at, now))
            conn.commit()
        self.log_audit("PAYMENT_ORDER", order_id, "STEP_UP_CHALLENGE_ISSUED",
                       {"challenge_id": challenge_id, "expires_at": expires_at})

    def count_step_up_challenges(self, order_id: str) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM step_up_challenges WHERE order_id = ?",
                                (order_id,)).fetchone()[0]

    def get_active_step_up_challenge(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("""
            SELECT * FROM step_up_challenges WHERE order_id = ? AND status = 'PENDING'
            ORDER BY created_at DESC LIMIT 1
            """, (order_id,)).fetchone()
            return dict(row) if row else None

    def record_step_up_attempt(self, challenge_id: str) -> int:
        """Atomically counts an attempt on a pending challenge; returns the new attempt count."""
        with self._get_connection() as conn:
            conn.execute("UPDATE step_up_challenges SET attempts = attempts + 1 "
                         "WHERE challenge_id = ? AND status = 'PENDING'", (challenge_id,))
            conn.commit()
            row = conn.execute("SELECT attempts FROM step_up_challenges WHERE challenge_id = ?",
                               (challenge_id,)).fetchone()
            return row[0] if row else 0

    def close_step_up_challenge(self, challenge_id: str, status: str) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE step_up_challenges SET status = ? WHERE challenge_id = ? AND status = 'PENDING'",
                         (status, challenge_id))
            conn.commit()

    def set_step_up_result(self, order_id: str, result: str, detail: str) -> bool:
        """PENDING -> PASSED | FAILED, exactly once."""
        with self._get_connection() as conn:
            cur = conn.execute("""
            UPDATE payment_orders SET step_up_status = ?, updated_at = ?
            WHERE order_id = ? AND risk_action = 'STEP_UP' AND step_up_status = 'PENDING'
            """, (result, datetime.now(timezone.utc).isoformat(), order_id))
            conn.commit()
            ok = cur.rowcount == 1
        if ok:
            self.log_audit("PAYMENT_ORDER", order_id, f"STEP_UP_{result}", {"detail": detail})
        return ok

    # --- Audit Log API ---
    def log_audit(self, entity_type: str, entity_id: str, action: str, payload: Any):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO audit_logs (entity_type, entity_id, action, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """, (
                entity_type,
                entity_id,
                action,
                json.dumps(payload) if isinstance(payload, (dict, list)) else str(payload),
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

    def list_audit_for(self, entity_ids: List[str], limit: int = 200) -> List[Dict[str, Any]]:
        """Audit entries for specific entities (alert, order, transaction), oldest first."""
        ids = [i for i in entity_ids if i]
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._get_connection() as conn:
            rows = conn.execute(f"SELECT * FROM audit_logs WHERE entity_id IN ({marks}) "
                                f"ORDER BY created_at ASC, log_id ASC LIMIT ?", (*ids, limit)).fetchall()
            return [dict(r) for r in rows]

    def list_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]


# Singleton instance
db = Database()
