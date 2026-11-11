"""
Phase 8: full-stack integration test.

Starts the real FastAPI backend (uvicorn, in-process, random port, temporary database, PayPal in
MOCK mode, Gemini disabled) and drives the real Streamlit dashboard through every demo flow:
  A  LOW    -> capture
  B  MEDIUM -> simulated OTP step-up -> capture
  C  HIGH   -> hold -> analyst approves (analyst key) -> capture -> case report PDF
plus the wallet-cluster graph tab.
"""

import os
import re
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from backend.app.main import app
from backend.app.database import Database
from backend.app.api import payments, fraud, webhooks
from backend.app.investigation_agent import task_manager as task_manager_module
from tests.keys import ANALYST_KEYS, MERCHANT_KEY

DASHBOARD = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
SCENARIOS = {"A": "Scenario A: Normal Payment (Clean Merchant)",
             "B": "Scenario B: Unusual Velocity / Device (Step-Up)",
             "C": "Scenario C: PYUSD Circular Ring (Hold & Investigate)"}


@pytest.fixture(scope="module")
def backend_url():
    mp = pytest.MonkeyPatch()
    db = Database(db_path=Path(tempfile.mkdtemp()) / "e2e.db")
    for module in (payments, fraud, webhooks, task_manager_module):
        mp.setattr(module, "db", db)
    mp.setattr(payments.paypal_client, "is_configured", False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    mp.setenv("PAYGUARD_API_BASE", url)
    mp.setenv("MERCHANT_API_KEY", MERCHANT_KEY)  # the dashboard plays the merchant backend
    yield url
    server.should_exit = True
    thread.join(timeout=5)
    mp.undo()


def btn(at, prefix):
    return next(b for b in at.button if b.label.startswith(prefix))


def msgs(at):
    return [e.value for kind in ("success", "warning", "error", "info") for e in getattr(at, kind)]


def start_payment(scenario):
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    at.sidebar.radio[0].set_value(SCENARIOS[scenario]).run()
    btn(at, "🧾").click().run()          # create PayPal order (mock)
    btn(at, "🔄").click().run()          # buyer approval (simulated in mock mode)
    btn(at, "⚡").click().run()          # pre-capture risk analysis
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_scenario_a_low_risk_is_captured(backend_url):
    at = start_payment("A")
    assert any("LOW risk" in m for m in msgs(at))
    btn(at, "💸").click().run()
    assert any("Captured (MOCK" in m for m in msgs(at))


def test_scenario_b_step_up_then_capture(backend_url):
    at = start_payment("B")
    assert any("MEDIUM risk" in m for m in msgs(at))
    btn(at, "📲").click().run()
    code = re.search(r"`(\d{6})`", next(m for m in msgs(at) if "SIMULATED delivery" in m)).group(1)
    order_id = at.query_params["order_id"]
    order_id = order_id[0] if isinstance(order_id, list) else order_id
    at.text_input(key=f"otp_input_{order_id}").set_value(code).run()
    btn(at, "✔️").click().run()
    assert any("Step-up verification passed" in m for m in msgs(at))
    btn(at, "💸").click().run()
    assert any("Captured (MOCK" in m for m in msgs(at))


def test_scenario_c_hold_analyst_approval_capture_and_case_report(backend_url):
    at = start_payment("C")
    assert any("HIGH risk" in m for m in msgs(at))
    btn(at, "Try capture").click().run()
    assert any("refused by server (403)" in m for m in msgs(at))

    alert_buttons = [b for b in at.button if b.key and b.key.startswith("app_")]
    alert_id = alert_buttons[-1].key[len("app_"):]
    at.text_input(key="analyst_key").set_value(ANALYST_KEYS["analyst.priya"]).run()
    at.text_input(key=f"notes_{alert_id}").set_value("Buyer verified via trusted channel.").run()
    at.button(key=f"app_{alert_id}").click().run()
    assert any("Captured" in m and "MOCK" in m for m in msgs(at))

    at.button(key=f"prep_{alert_id}").click().run()
    pdf = at.session_state[f"case_pdf_{alert_id}"]
    assert pdf.startswith(b"%PDF") and len(pdf) > 3000
    assert not at.exception


def test_graph_tab_shows_computed_ring_evidence(backend_url):
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    at.selectbox[0].set_value("cluster_ring_904").run()
    assert any("CIRCULAR_TRANSFER_CYCLE" in e.value for e in at.error)
    assert len(at.dataframe) >= 1 and not at.exception
