"""
Layer 5/6: "PayGuard AI - Suspicious Activity Case Report" (PDF).

Built ONLY from records PayGuard has stored for the case: the alert (analyst decision, AI report,
wallet-cluster evidence), the risk evaluation (score breakdown, rule factors, SHAP, signal
availability, server-computed behaviour profile), the PayPal order (status, capture) and the
audit trail. Nothing is recomputed or inferred at export time.

This is an INTERNAL investigation record from a prototype. It is NOT a Suspicious Activity
Report (SAR) filed with any regulator, and says so on every page.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

from ..database import Database

TITLE = "PayGuard AI - Suspicious Activity Case Report"
DISCLAIMER = ("Internal investigation record generated from PayGuard's stored evidence (prototype; PayPal Sandbox "
              "and synthetic/public data). NOT a Suspicious Activity Report (SAR) filed with any regulator. "
              "Indicators are investigative leads, not proof of fraud.")

_ASCII = {"—": "-", "–": "-", "→": "->", "←": "<-", "…": "...", "≥": ">=", "≤": "<=", "“": '"', "”": '"',
          "‘": "'", "’": "'", "•": "-", "×": "x", "✓": "v", "🔺": "+", "🔻": "-", "·": "-", " ": " "}
# Audit actions whose payloads are summarised (other payloads are omitted: they can be large or detailed).
_AUDIT_SUMMARY_FIELDS = {"analyst_id", "notes", "capture_id", "paypal_status", "llm_status",
                         "evidence_categories_sent_to_llm", "risk_band", "risk_action", "approval_source",
                         "detail", "by", "error", "fallback_reason"}


def _t(value: Any) -> str:
    """Text safe for the PDF core fonts (latin-1)."""
    text = "" if value is None else str(value)
    for k, v in _ASCII.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


# ------------------------------------------------------------------ data gathering
def gather_case(db: Database, alert_id: str) -> Optional[Dict[str, Any]]:
    alert = db.get_alert(alert_id)
    if not alert:
        return None
    tx = db.get_transaction(alert["transaction_id"]) or {}
    order = db.get_payment_order_by_transaction(alert["transaction_id"])
    audit = db.list_audit_for([alert_id, alert["transaction_id"], order["order_id"] if order else None])
    return {"alert": alert, "transaction": tx, "order": order, "audit": audit}


def _audit_summary(entry: Dict[str, Any]) -> str:
    import json
    try:
        payload = json.loads(entry.get("payload") or "{}")
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    parts = [f"{k}={v}" for k, v in payload.items() if k in _AUDIT_SUMMARY_FIELDS and v not in (None, "", [])]
    return "; ".join(parts)[:160]


# ------------------------------------------------------------------ PDF rendering
class _ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(0, 48, 135)
        self.cell(0, 6, _t(TITLE), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(0, 112, 186)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(110, 110, 110)
        self.multi_cell(0, 3.2, _t(f"{DISCLAIMER}  Page {self.page_no()}/{{nb}}"), align="C")


def _h(pdf: FPDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 48, 135)
    pdf.cell(0, 7, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


def _p(pdf: FPDF, text: str, size: float = 9, style: str = "") -> None:
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, 4.6, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _kv(pdf: FPDF, rows: List[tuple]) -> None:
    pdf.set_font("Helvetica", "", 8.5)
    with pdf.table(col_widths=(62, 118), first_row_as_headings=False, line_height=4.6,
                   borders_layout="HORIZONTAL_LINES") as table:
        for key, value in rows:
            r = table.row()
            r.cell(_t(key))
            r.cell(_t(value if value not in (None, "") else "-"))


def _grid(pdf: FPDF, headings: List[str], rows: List[List[Any]], widths: tuple, size: float = 7.5) -> None:
    if not rows:
        _p(pdf, "None recorded.", 8.5, "I")
        return
    pdf.set_font("Helvetica", "", size)
    with pdf.table(col_widths=widths, line_height=4.2,
                   headings_style=FontFace(emphasis="BOLD", fill_color=(230, 238, 248))) as table:
        h = table.row()
        for cell in headings:
            h.cell(_t(cell))
        for row in rows:
            r = table.row()
            for cell in row:
                r.cell(_t(cell if cell not in (None, "") else "-"))


def _addr(a: Any) -> str:
    """Abbreviated address for narrow table cells (full values are in section 3 / the cluster API)."""
    a = str(a or "")
    return f"{a[:8]}...{a[-6:]}" if len(a) > 18 else a


def render_case_pdf(case: Dict[str, Any], exported_by: str) -> bytes:
    alert, tx, order, audit = case["alert"], case["transaction"], case["order"], case["audit"]
    raw = tx.get("raw_payload") or {}
    prof = raw.get("server_computed_profile") or {}
    b = tx.get("score_breakdown") or {}
    cf = alert.get("cluster_findings") or {}
    rep = alert.get("llm_report") or {}

    pdf = _ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 12, 15)
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _t(TITLE), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _p(pdf, DISCLAIMER, 8, "I")

    _h(pdf, "1. Case identification")
    _kv(pdf, [
        ("Case ID", alert.get("alert_id")),
        ("Transaction reference", alert.get("transaction_id")),
        ("PayPal order ID", order.get("order_id") if order else "No linked PayPal order"),
        ("Payment mode", ("MOCK (simulated, no PayPal call)" if order and order.get("mock_mode")
                          else "PayPal Sandbox" if order else "-")),
        ("Alert created (UTC)", alert.get("created_at")),
        ("Report generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Exported by", exported_by),
    ])

    _h(pdf, "2. Risk score and category")
    model_part = (f"model {b.get('model_points')} (P(fraud) {b['model_probability']:.1%})"
                  if b.get("model_probability") is not None else "model unavailable")
    _kv(pdf, [
        ("Risk score / band", f"{tx.get('risk_score', alert.get('risk_score'))}/100 - {tx.get('risk_band', alert.get('risk_band'))}"),
        ("Required action", tx.get("action")),
        ("Score breakdown", (f"base {b.get('base_points')} + {model_part} + rules {b.get('rule_points')} + "
                             f"wallet {b.get('wallet_points')} = {b.get('raw_total')} -> {b.get('final_score')}")
                            if b else "-"),
        ("Signal availability", ", ".join(f"{k}: {v}" for k, v in (tx.get("signal_availability") or {}).items())),
        ("Policy overrides", "; ".join(tx.get("policy_overrides") or []) or "None"),
        ("Decision latency", f"{tx.get('decision_latency_ms')} ms"),
    ])

    _h(pdf, "3. Transaction summary")
    _kv(pdf, [
        ("Amount", f"{float(tx.get('amount', alert.get('amount')) or 0):,.2f} {tx.get('currency', alert.get('currency'))}"),
        ("Sender / recipient (demo labels)", f"{tx.get('sender_id')} -> {tx.get('recipient_id')}"),
        ("Corridor", f"{raw.get('sender_country', '-')} -> {raw.get('recipient_country', '-')}"),
        ("Linked recipient wallet", tx.get("recipient_wallet") or "None provided"),
        ("Evaluated at (UTC)", tx.get("created_at")),
        ("Order status / PayPal status", f"{order.get('status')} / {order.get('paypal_status')}" if order else "-"),
        ("Buyer approval", f"{order.get('approval_source')} at {order.get('approved_at')}" if order and order.get("approved_at") else "-"),
        ("Step-up verification", order.get("step_up_status") if order else "-"),
    ])

    _h(pdf, "4. Behavioural findings (server-computed)")
    if prof:
        _kv(pdf, [
            ("Sender baseline", prof.get("baseline_status")),
            ("Completed payments in history", f"{prof.get('history_count')} ({prof.get('seed_count')} synthetic seed, "
                                              f"{prof.get('stored_count')} stored)"),
            ("Average completed payment", prof.get("avg_amount")),
            ("Attempts in last 24 h", prof.get("attempts_last_24h")),
            ("Device seen before (simulated ID)", "yes" if prof.get("device_seen_before") else "no"),
            ("Recipient seen before", "yes" if prof.get("recipient_seen_before") else "no"),
            ("Recipient's prior received payments", prof.get("recipient_prior_receipts")),
        ])
    else:
        _p(pdf, "No behaviour profile stored for this evaluation.", 8.5, "I")

    _h(pdf, "5. Wallet intelligence and suspicious cluster evidence")
    _kv(pdf, [("Wallet intelligence", (tx.get("signal_availability") or {}).get("wallet_intelligence")),
              ("Cluster finding", f"{cf.get('pattern_type')} ({cf.get('cluster_id')})" if cf else "No indicator found"),
              ("Details", cf.get("details") if cf else "-"),
              ("Time window (UTC)", f"{cf.get('time_window_start')} -> {cf.get('time_window_end')}" if cf else "-"),
              ("Data provenance", ", ".join(cf.get("provenance") or []) if cf else "-")])
    if cf.get("evidence_transfers"):
        pdf.ln(1)
        _grid(pdf, ["Transfer", "Time (UTC)", "From", "To", "PYUSD", "Tx hash"],
              [[t.get("transfer_id"), t.get("timestamp"), _addr(t.get("from")), _addr(t.get("to")),
                f"{t.get('amount_pyusd', 0):,.2f}", _addr(t.get("tx_hash")) if t.get("tx_hash") else "(synthetic)"]
               for t in cf["evidence_transfers"]], (26, 36, 34, 34, 22, 28), size=7)
        _p(pdf, "Addresses abbreviated in this table; full values are stored with the case.", 7, "I")

    _h(pdf, "6. Model explanation - top SHAP contributions (XGBoost trained on PaySim, simulated data)")
    _grid(pdf, ["Feature", "Value", "SHAP (log-odds)", "Effect"],
          [[e.get("label"), e.get("value"), f"{e.get('shap_value', 0):+.3f}", e.get("direction")]
           for e in (tx.get("model_explanations") or [])[:3]], (70, 45, 30, 35))
    _h(pdf, "7. Rule factors (fixed policy weights, not learned)")
    _grid(pdf, ["Factor", "Points", "Description"],
          [[f.get("factor"), f"{f.get('impact', 0):+g}", f.get("description")] for f in (tx.get("contributing_factors") or [])],
          (55, 15, 110))

    _h(pdf, "8. AI investigation report")
    if rep and rep.get("llm_status") in ("GEMINI", "FALLBACK"):
        _p(pdf, f"Source: {rep.get('generated_by')}" + (f" (Gemini not used: {rep.get('fallback_reason')})"
                                                         if rep.get("llm_status") == "FALLBACK" else "")
                + f"; generated {rep.get('generated_at')}.", 8, "I")
        _p(pdf, "Executive summary", 9, "B")
        _p(pdf, rep.get("executive_summary", ""))
        _p(pdf, "Red flags (with cited evidence)", 9, "B")
        for f in rep.get("top_red_flags") or []:
            if isinstance(f, dict):
                _p(pdf, f"- {f.get('text')} [{', '.join(f.get('evidence_ids') or [])}]")
            else:
                _p(pdf, f"- {f}")
        _p(pdf, "Suggested analyst action (advisory only)", 9, "B")
        _p(pdf, f"{rep.get('recommended_action_label') or rep.get('recommended_action')}"
                + (f" - {rep.get('action_rationale')}" if rep.get("action_rationale") else ""))
        if rep.get("evidence"):
            _p(pdf, f"Evidence items (categories sent to the LLM: "
                    f"{', '.join(rep.get('evidence_categories_sent') or []) or 'none'})", 9, "B")
            _grid(pdf, ["ID", "Category", "Evidence"],
                  [[e["id"], e["category"], e["text"]] for e in rep["evidence"]], (10, 32, 138), size=7)
    else:
        _p(pdf, "No AI investigation report stored for this case.", 8.5, "I")

    _h(pdf, "9. Recommended analyst action and analyst decision")
    _kv(pdf, [
        ("Suggested action (advisory)", rep.get("recommended_action_label") or "-"),
        ("Alert status", alert.get("status")),
        ("Analyst decision", alert.get("analyst_decision") or "Pending"),
        ("Decided by (from API key)", alert.get("analyst_id")),
        ("Decided at (UTC)", alert.get("decided_at")),
        ("Reason", alert.get("analyst_notes")),
        ("Capture outcome", f"{alert.get('capture_status') or '-'}  {alert.get('capture_detail') or ''}"),
    ])

    _h(pdf, "10. Audit trail")
    _grid(pdf, ["Time (UTC)", "Entity", "Action", "Summary"],
          [[a.get("created_at", "")[:19], a.get("entity_type"), a.get("action"), _audit_summary(a)] for a in audit],
          (32, 28, 50, 70), size=6.8)

    return bytes(pdf.output())
