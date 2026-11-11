"""
Phase 7: "PayGuard AI - Suspicious Activity Case Report" PDF export.
The PDF text is extracted with pypdf and checked against the stored case data.
"""

import io
import tempfile
import uuid
from pathlib import Path

import pypdf
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module
from backend.app.investigation_agent.case_report import _t
from tests.keys import MERCHANT_HEADERS, analyst_headers

RING = "0xde5a020000000000000000000000000000000001"
HIGH = dict(sender_id="user_buyer_03", recipient_id="p2p_crypto_seller_99", amount=4800.0, currency="USD",
            recipient_wallet=RING, device_id="vpn_tor_exit_node_12", sender_country="US", recipient_country="KY")


@pytest.fixture
def env(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = Path(tf.name)
    test_db = Database(db_path=path)
    for module in (payments, fraud, webhooks, task_manager_module):
        monkeypatch.setattr(module, "db", test_db)
    monkeypatch.setattr(payments.paypal_client, "is_configured", False)
    return TestClient(app, headers=MERCHANT_HEADERS), test_db


def held_case(client, with_order=True):
    tx = f"TX-{uuid.uuid4().hex[:8]}"
    body = {"transaction_id": tx, **HIGH}
    if with_order:
        oid = client.post("/payments/create", json={"amount": 4800.0, "recipient_email": "m@example.com"}).json()["order_id"]
        client.post(f"/payments/{oid}/verify-approval")
        body["order_id"] = oid
    assert client.post("/fraud/analyze", json=body).json()["risk_band"] == "HIGH"
    return f"ALERT-{tx}", tx


def pdf_text(content: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(content))
    # Normalise whitespace: PDF line wrapping can split a phrase across lines.
    return " ".join(" ".join(page.extract_text() for page in reader.pages).split())


def download(client, alert_id, headers=None):
    return client.get(f"/alerts/{alert_id}/case-report.pdf", headers=headers or analyst_headers())


# --- Content ---------------------------------------------------------------------

def test_report_contains_every_required_section(env):
    client, _ = env
    alert_id, tx = held_case(client)
    client.post(f"/alerts/{alert_id}/review", headers=analyst_headers("analyst.rahul"),
                json={"analyst_decision": "REJECT", "analyst_notes": "Fast circular PYUSD loop."})
    resp = download(client, alert_id)
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")
    assert f'payguard_case_{alert_id}.pdf' in resp.headers["content-disposition"]

    text = pdf_text(resp.content)
    for expected in [
        "PayGuard AI - Suspicious Activity Case Report",
        "NOT a Suspicious Activity Report (SAR) filed with any regulator",
        alert_id, tx,                                                   # case id, transaction reference
        "100/100 - HIGH", "Score breakdown",                            # risk score and category
        "4,800.00 USD", "US -> KY",                                     # transaction summary
        "Sender baseline", "Average completed payment",                 # behavioural findings
        "CIRCULAR_TRANSFER_CYCLE", "SYN-RING-001", "(synthetic)",       # cluster evidence
        "SHAP", "Recipient's prior received payments",                  # top SHAP
        "Transaction Rule: Amount Spike",                               # rule factors
        "AI investigation report", "Executive summary", "[E1]",         # AI report (fallback in tests)
        "Suggested analyst action (advisory only)",                     # recommendation
        "REJECT", "analyst.rahul", "Fast circular PYUSD loop.",         # analyst decision, identity, reason
        "BLOCKED_BY_ANALYST",                                           # outcome
        "Audit trail", "RISK_DECISION_ATTACHED", "ANALYST_REJECT",      # audit
    ]:
        assert expected in text, expected


def test_pending_case_without_order_is_reported_honestly(env):
    client, _ = env
    alert_id, _ = held_case(client, with_order=False)
    text = pdf_text(download(client, alert_id).content)
    assert "No linked PayPal order" in text and "Pending" in text


def test_stored_gemini_report_is_rendered_with_citations(env):
    client, db = env
    alert_id, _ = held_case(client)
    db.update_alert_llm_report(alert_id, {
        "llm_status": "GEMINI", "generated_by": "Gemini (gemini-3.5-flash)", "generated_at": "2026-11-09T00:00:00Z",
        "executive_summary": "Possible layering indicator on the recipient wallet.",
        "top_red_flags": [{"text": "Loop of 5 wallets in 4.5 hours", "evidence_ids": ["E1"]}],
        "recommended_action": "ESCALATE_ENHANCED_DUE_DILIGENCE",
        "recommended_action_label": "Escalate for enhanced due diligence before any capture decision",
        "evidence": [{"id": "E1", "category": "wallet_cluster", "text": "loop"}],
        "evidence_categories_sent": ["wallet_cluster"]})
    text = pdf_text(download(client, alert_id).content)
    assert "Gemini (gemini-3.5-flash)" in text and "Loop of 5 wallets in 4.5 hours [E1]" in text
    assert "categories sent to the LLM: wallet_cluster" in text


def test_unicode_text_does_not_break_the_pdf(env):
    client, _ = env
    alert_id, _ = held_case(client)
    client.post(f"/alerts/{alert_id}/review", headers=analyst_headers(),
                json={"analyst_decision": "REJECT", "analyst_notes": "Café → ring — confirmed ✓ 🔺"})
    resp = download(client, alert_id)
    assert resp.status_code == 200 and "Caf" in pdf_text(resp.content)
    assert _t("a → b — c ≥ d") == "a -> b - c >= d"


# --- Access and audit -------------------------------------------------------------

def test_export_requires_a_key(env):
    client, _ = env
    alert_id, _ = held_case(client)
    assert client.get(f"/alerts/{alert_id}/case-report.pdf", headers={"X-API-Key": ""}).status_code == 401
    assert download(client, alert_id, MERCHANT_HEADERS).status_code == 200


@pytest.mark.parametrize("alert_id,code", [("ALERT-DOES-NOT-EXIST", 404), ("bad id!", 400)])
def test_export_errors(env, alert_id, code):
    client, _ = env
    assert download(client, alert_id).status_code == code


def test_export_is_audited_with_exporter_identity(env):
    client, db = env
    alert_id, _ = held_case(client)
    download(client, alert_id, analyst_headers("analyst.priya"))
    entries = [a for a in db.list_audit_for([alert_id]) if a["action"] == "CASE_REPORT_EXPORTED"]
    assert entries and "analyst:analyst.priya" in entries[0]["payload"]
