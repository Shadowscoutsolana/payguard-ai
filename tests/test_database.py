"""
Unit tests for PayGuard AI SQLite Storage Layer (database.py).
"""

import pytest
import tempfile
from pathlib import Path
from backend.app.database import Database


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = Path(tf.name)
    db = Database(db_path=db_path)
    yield db
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


def test_save_and_get_transaction(temp_db):
    tx_data = {
        "transaction_id": "TX-TEST-001",
        "sender_id": "user_buyer_01",
        "recipient_id": "merchant_tech",
        "amount": 45.0,
        "currency": "USD",
        "risk_score": 15,
        "risk_band": "LOW",
        "action": "CAPTURE",
        "decision_latency_ms": 4.2,
        "reasons": ["Standard parameters"],
        "contributing_factors": [{"factor": "Baseline", "impact": 10.0, "description": "Normal"}],
        "raw_payload": {"test": True}
    }
    temp_db.save_transaction(tx_data)
    retrieved = temp_db.get_transaction("TX-TEST-001")

    assert retrieved is not None
    assert retrieved["transaction_id"] == "TX-TEST-001"
    assert retrieved["amount"] == 45.0
    assert retrieved["risk_score"] == 15
    assert retrieved["reasons"] == ["Standard parameters"]


def test_alert_lifecycle(temp_db):
    alert = {
        "alert_id": "ALERT-TX-999",
        "transaction_id": "TX-999",
        "status": "PENDING_REVIEW",
        "risk_score": 88,
        "risk_band": "HIGH",
        "amount": 4800.0,
        "currency": "USD",
        "recipient_id": "crypto_seller_99",
        "reasons": ["Circular transfer cycle detected"]
    }
    temp_db.save_alert(alert)

    # Check alert listed
    alerts = temp_db.list_alerts()
    assert len(alerts) == 1
    assert alerts[0]["alert_id"] == "ALERT-TX-999"
    assert alerts[0]["status"] == "PENDING_REVIEW"

    # Resolve alert
    updated = temp_db.resolve_alert("ALERT-TX-999", "REJECT", "Confirmed mule wallet loop.")
    assert updated["status"] == "RESOLVED"
    assert updated["analyst_decision"] == "REJECT"
    assert updated["analyst_notes"] == "Confirmed mule wallet loop."


def test_payment_events_and_audit(temp_db):
    temp_db.log_payment_event(
        event_id="EVT-01",
        order_id="ORDER-123",
        event_type="CHECKOUT.ORDER.APPROVED",
        status="APPROVED",
        details={"buyer": "test@example.com"}
    )
    events = temp_db.list_payment_events()
    assert len(events) == 1
    assert events[0]["event_id"] == "EVT-01"

    logs = temp_db.list_audit_logs()
    assert len(logs) >= 1
