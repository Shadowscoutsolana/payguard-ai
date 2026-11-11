"""
PayGuard AI - Streamlit Fraud Investigation & Risk Operations Dashboard.
Visualizes real-time pre-capture transactions, interactive PyVis PYUSD wallet graphs,
decision latency benchmarks, actionable AI-generated incident reports, and PayPal webhook monitors.
"""

import streamlit as st
import requests
import json
import os
import time
from pathlib import Path

# Import our PyVis interactive graph generator
from graph_viz import generate_interactive_graph_html

# Page Configuration
st.set_page_config(
    page_title="PayGuard AI | Risk Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 127.0.0.1, not "localhost": on Windows, Python resolves localhost to IPv6 ::1 first and
# waits ~2 s for the refusal before falling back to IPv4 (uvicorn listens on IPv4 only).
API_BASE = os.getenv("PAYGUARD_API_BASE", "http://127.0.0.1:8000")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_wallets.json"

# The dashboard plays the MERCHANT backend in this demo, so it holds the merchant key
# (read server-side from the environment / .env; never shown in the page).
from dotenv import dotenv_values
MERCHANT_API_KEY = os.getenv("MERCHANT_API_KEY") or dotenv_values(Path(__file__).resolve().parent.parent / ".env").get("MERCHANT_API_KEY", "")
MERCHANT_HEADERS = {"X-API-Key": MERCHANT_API_KEY or ""}

# Header & Branding
st.markdown("""
<div style="background: linear-gradient(90deg, #003087 0%, #0079C1 100%); padding: 1.4rem; border-radius: 10px; margin-bottom: 1.2rem; color: white;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">🛡️ PayGuard AI</h1>
            <p style="margin: 0.2rem 0 0 0; font-size: 1.05rem; opacity: 0.92;">
                Pre-Capture Risk Gate, PYUSD Graph Intelligence & Actionable GenAI Investigation
            </p>
        </div>
        <div style="text-align: right; background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 8px;">
            <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Pre-Capture Gate SLA</span><br/>
            <strong style="font-size: 1.3rem;">&lt; 100 ms Synchronous</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Demo Scenarios & Clusters from JSON
demo_data = {}
if DATA_PATH.exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        demo_data = json.load(f)

# Sidebar: Controls & Live Scenarios
st.sidebar.title("🎮 Hackathon Demo Controls")
st.sidebar.markdown("Run the 3 pre-scripted demo scenarios directly through the Pre-Capture Risk Gate:")

scenario_options = [
    "Scenario A: Normal Payment (Clean Merchant)",
    "Scenario B: Unusual Velocity / Device (Step-Up)",
    "Scenario C: PYUSD Circular Ring (Hold & Investigate)"
]

# If a payment is in progress (e.g. after PayPal redirects back, which starts a fresh
# session), lock the sidebar to that order's own scenario so the screen stays consistent.
active_order_scenario = None
_active_order_id = st.query_params.get("order_id")
if _active_order_id and _active_order_id != "invalid":
    try:
        _r = requests.get(f"{API_BASE}/payments/{_active_order_id}", headers=MERCHANT_HEADERS, timeout=5.0)
        if _r.status_code == 200:
            active_order_scenario = (_r.json().get("checkout_context") or {}).get("demo_scenario")
    except requests.RequestException:
        pass

if active_order_scenario in scenario_options:
    demo_scenario = st.sidebar.radio("Select Scenario:", scenario_options,
                                     index=scenario_options.index(active_order_scenario),
                                     disabled=True, key="scenario_locked")
    st.sidebar.caption("🔒 Locked to the active order. Click **Start a new payment** to change.")
else:
    demo_scenario = st.sidebar.radio("Select Scenario:", scenario_options, key="scenario_free")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")
try:
    health_resp = requests.get(f"{API_BASE}/", timeout=1.0)
    if health_resp.status_code == 200:
        st.sidebar.success("🟢 Risk Engine Online (<100ms SLA)")
    else:
        st.sidebar.warning("🟡 Risk Engine Warning")
except Exception:
    st.sidebar.info("ℹ️ Local Gateway Mode (Mock Ready)")

st.sidebar.markdown("""
---
**🎯 Graph Node Legend:**
* <span style="color:#00d2ff; font-weight:bold;">🔵 Blue Diamond</span>: Current Payment Recipient
* <span style="color:#FF5252; font-weight:bold;">🔴 Red Node</span>: In a computed suspicious-pattern indicator
* <span style="color:#69F0AE; font-weight:bold;">🟢 Green Node</span>: Allowlisted merchant (demo)
* <span style="color:#BDBDBD; font-weight:bold;">⚪ Grey Node</span>: No indicator found (not proof of being clean)
""", unsafe_allow_html=True)

# Main Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 PayPal Checkout + Risk Gate",
    "🕸️ Interactive Graph Visualization",
    "📋 Actionable AI Investigation Queue",
    "🔔 PayPal Webhooks & Compliance"
])

# ----------------- TAB 1: END-TO-END PAYPAL CHECKOUT WITH PRE-CAPTURE GATE -----------------
# Flow: create PayPal order -> buyer approves on PayPal -> server verifies approval
#       -> pre-capture risk gate -> capture / step-up / hold (all enforced server-side).
# The order id lives in the URL (?order_id=...) because PayPal's redirect back to the
# dashboard starts a fresh Streamlit session; everything else is reloaded from the backend.

SCENARIO_INPUTS = {
    "A": dict(amount=45.0, sender_id="user_buyer_01", recipient_id="merchant_tech_supplies",
              recipient_wallet="0xde5a010000000000000000000000000000000001", device_id="known_macbook_pro_01",
              sender_country="US", recipient_country="US"),
    "B": dict(amount=950.0, sender_id="user_buyer_02", recipient_id="new_digital_giftcard_store",
              recipient_wallet="", device_id="unrecognized_linux_vm_88",
              sender_country="US", recipient_country="SG"),
    "C": dict(amount=4800.0, sender_id="user_buyer_03", recipient_id="p2p_crypto_seller_99",
              recipient_wallet="0xde5a020000000000000000000000000000000001", device_id="vpn_tor_exit_node_12",
              sender_country="US", recipient_country="KY"),
}

FLOW_STEPS = ["Order created", "Buyer approved", "Risk evaluated", "Captured"]
STEP_INDEX = {"CREATED": 0, "APPROVED": 1, "RISK_EVALUATED": 2, "CAPTURE_IN_PROGRESS": 2,
              "CAPTURE_FAILED": 2, "CAPTURED": 3, "CANCELLED": 0}


def md(text):
    """Escape '$' so Streamlit does not render text between two dollar amounts as a LaTeX formula."""
    return str(text if text is not None else "").replace("$", r"\$")


def api_call(method, path, headers=None, **kwargs):
    """Returns (status_code, json_body). status_code is None if the backend is unreachable.
    Sends the merchant key unless other headers (e.g. an analyst key) are given."""
    try:
        resp = requests.request(method, f"{API_BASE}{path}", timeout=15.0,
                                headers=headers if headers is not None else MERCHANT_HEADERS, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        return resp.status_code, body
    except requests.RequestException as exc:
        return None, {"detail": f"Backend unreachable at {API_BASE}: {exc}"}


def server_message(body):
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, list):  # FastAPI validation errors -> plain sentences
        return "; ".join(f"{(e.get('loc') or ['?'])[-1]}: {e.get('msg')}" for e in detail if isinstance(e, dict))
    return str(detail)


def start_new_payment():
    st.query_params.clear()
    for key in ("last_capture", "last_action_error"):
        st.session_state.pop(key, None)


def render_risk_result(tx):
    band, score = tx.get("risk_band"), tx.get("risk_score")
    if band == "LOW":
        st.success(f"**LOW risk — {score}/100.** Eligible for capture.")
    elif band == "MEDIUM":
        st.warning(f"**MEDIUM risk — {score}/100.** Step-up verification required before capture.")
    else:
        st.error(f"**HIGH risk — {score}/100.** Payment held for analyst review.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk engine latency", f"{tx.get('decision_latency_ms', 0):.1f} ms")
    m2.metric("Required action", tx.get("action", "—"))
    wallet_signal = (tx.get("signal_availability") or {}).get("wallet_intelligence", "—")
    m3.metric("Wallet intelligence", wallet_signal.replace("_", " "))
    for override in tx.get("policy_overrides") or []:
        st.warning(f"**Policy override:** {md(override)}")
    prof = (tx.get("raw_payload") or {}).get("server_computed_profile")
    if prof:
        avg = f"${prof['avg_amount']:.2f}" if prof.get("avg_amount") is not None else "n/a"
        st.caption(
            f"**Sender baseline (server-computed):** {prof['baseline_status']} · "
            f"{prof['history_count']} completed payments ({prof['seed_count']} synthetic seed, "
            f"{prof['stored_count']} stored) · avg {avg} · {prof['attempts_last_24h']} attempts in last 24h · "
            f"device seen before: {'yes' if prof['device_seen_before'] else 'no'} · "
            f"recipient seen before: {'yes' if prof['recipient_seen_before'] else 'no'}"
        )
    b = tx.get("score_breakdown") or {}
    if b:
        model_part = (f"model {b['model_points']:g} (P(fraud) {b['model_probability']:.1%})"
                      if b.get("model_probability") is not None else "model UNAVAILABLE")
        st.markdown(f"**Score breakdown:** base {b['base_points']} + {model_part} + rules {b['rule_points']:g} "
                    f"+ wallet {b['wallet_points']:g} = {b['raw_total']:g} → **{b['final_score']}** (clipped to 0–100)")

    st.markdown("**🤖 Model explanation** — XGBoost trained on PaySim (simulated mobile-money data), "
                "top SHAP contributions in log-odds")
    if tx.get("model_explanations"):
        for e in tx["model_explanations"]:
            arrow = "🔺" if e["shap_value"] > 0 else "🔻"
            st.markdown(f"- {arrow} **{e['label']}** = {md(e['value'])} → {e['direction']} (SHAP {e['shap_value']:+.2f})")
    else:
        st.caption("No model explanation (model unavailable for this evaluation).")

    st.markdown("**📏 Rule factors** — fixed policy weights, not learned (points actually added to the score)")
    if tx.get("contributing_factors"):
        for factor in tx["contributing_factors"]:
            st.info(f"**{factor['factor']}** ({factor['impact']:+g} pts): {md(factor['description'])}")
    else:
        st.caption("No rule factors triggered.")


def render_step_up(order_id, step_up_status):
    """Simulated OTP step-up for MEDIUM-risk orders. All checks happen on the server."""
    if step_up_status == "PASSED":
        st.success("✅ Step-up verification passed (simulated OTP). Capture is now permitted.")
        return
    if step_up_status == "FAILED":
        st.error("❌ Step-up verification failed — escalated to analyst review (see the Investigation Queue tab).")
        return

    st.warning("**MEDIUM risk:** the buyer must pass step-up verification before capture.")
    otp_info = st.session_state.get(f"otp_{order_id}")
    if st.button("📲 Send verification code (simulated)" if not otp_info else "📲 Send a new code"):
        code, body = api_call("POST", f"/payments/{order_id}/step-up/start")
        if code == 200:
            st.session_state[f"otp_{order_id}"] = body
        else:
            st.session_state["last_action_error"] = f"Could not issue code ({code}): {server_message(body)}"
        st.rerun()

    if otp_info:
        if otp_info.get("simulated_code"):
            st.info(f"**SIMULATED delivery** — no SMS/email channel exists in this prototype, so the code is "
                    f"shown here: **`{otp_info['simulated_code']}`** · expires in "
                    f"{otp_info['expires_in_seconds'] // 60} min · {otp_info['max_attempts']} attempts")
        entered = st.text_input("Enter the 6-digit code", max_chars=6, key=f"otp_input_{order_id}")
        if st.button("✔️ Verify code"):
            code, body = api_call("POST", f"/payments/{order_id}/step-up/verify", json={"code": entered.strip()})
            if code != 200:
                st.session_state["last_action_error"] = f"Verification failed ({code}): {server_message(body)}"
            st.rerun()


with tab1:
    params = st.query_params
    order_id = params.get("order_id")

    if order_id in (None, "", "invalid"):
        if params.get("approval") == "FAILED":
            st.error(f"PayPal approval could not be verified: {params.get('reason', 'unknown error')}")
        # ---------- STEP 1: create a PayPal order ----------
        scenario_key = demo_scenario.split(":")[0].split()[-1]  # "A" / "B" / "C"
        defaults = SCENARIO_INPUTS[scenario_key]
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Step 1 — Create PayPal order")
            st.caption(f"Pre-filled from **{demo_scenario}** (sidebar). Values are demo inputs.")
            amount = st.number_input("Amount (USD)", value=float(defaults["amount"]), min_value=1.0,
                                     key=f"amt_{scenario_key}")
            sender_id = st.text_input("Sender ID", value=defaults["sender_id"], key=f"snd_{scenario_key}")
            recipient_id = st.text_input("Recipient ID", value=defaults["recipient_id"], key=f"rcp_{scenario_key}")
            recipient_wallet = st.text_input("Linked PYUSD / Ethereum wallet (demo association)",
                                             value=defaults["recipient_wallet"], key=f"wal_{scenario_key}")
            device_id = st.text_input("Device ID (simulated)", value=defaults["device_id"], key=f"dev_{scenario_key}")
            st.caption(f"Corridor: {defaults['sender_country']} → {defaults['recipient_country']} · "
                       "Payment history is computed by the server from completed payments "
                       "(demo personas start with a labelled synthetic seed).")
            create_btn = st.button("🧾 Create PayPal order", type="primary", width="stretch")
        with col2:
            st.subheader("How this demo works")
            st.markdown(
                "1. PayGuard creates a **PayPal Sandbox** order (intent `CAPTURE`).\n"
                "2. The buyer approves it on PayPal's own page.\n"
                "3. PayGuard **re-checks the approval with PayPal's API** — the redirect alone is not trusted.\n"
                "4. The pre-capture risk gate scores the payment.\n"
                "5. The server decides: **capture** (LOW), **step-up** (MEDIUM) or **hold** (HIGH)."
            )
            st.info("Use your **Sandbox Personal (buyer)** account on the PayPal page — not the seller account.")

        if create_btn:
            payload = {
                "amount": amount, "currency": "USD", "recipient_email": "merchant@payguard.example",
                "description": f"PayGuard demo — {demo_scenario.split(':')[0]}",
                "demo_scenario": demo_scenario,
                "sender_id": sender_id, "recipient_id": recipient_id,
                "recipient_wallet": recipient_wallet or None, "device_id": device_id,
                "sender_country": defaults["sender_country"], "recipient_country": defaults["recipient_country"],
            }
            code, body = api_call("POST", "/payments/create", json=payload)
            if code == 200:
                st.query_params["order_id"] = body["order_id"]
                st.rerun()
            else:
                st.error(f"Could not create order ({code}): {server_message(body)}")

    else:
        # ---------- An order is in progress: always render from server state ----------
        code, order = api_call("GET", f"/payments/{order_id}")
        if code != 200:
            st.error(f"Could not load order `{order_id}` ({code}): {server_message(order)}")
            st.button("Start a new payment", on_click=start_new_payment)
            st.stop()

        status = order["status"]
        ctx = order.get("checkout_context") or {}
        mode_label = "MOCK (simulated)" if order.get("mock_mode") else "PayPal Sandbox (live API)"

        head1, head2 = st.columns([3, 1])
        with head1:
            st.subheader(f"Order `{order_id}` — ${order['amount']:.2f} {order['currency']}")
            st.markdown(f"**{ctx.get('demo_scenario') or 'Custom payment (no demo scenario)'}**")
            st.caption(f"Mode: **{mode_label}** · Server status: **{status}** · "
                       f"Recipient: `{ctx.get('recipient_id')}` · Wallet: `{ctx.get('recipient_wallet') or 'none'}`")
        with head2:
            st.button("↺ Start a new payment", on_click=start_new_payment, width="stretch")

        step = STEP_INDEX.get(status, 0)
        st.progress((step + 1) / len(FLOW_STEPS),
                    text=" → ".join(f"**{s}**" if i <= step else s for i, s in enumerate(FLOW_STEPS)))

        if params.get("approval") == "FAILED":
            st.error(f"Approval check failed: {params.get('reason', '')}")

        if st.session_state.get("last_action_error"):
            st.error(st.session_state.pop("last_action_error"))

        # ---------- STEP 2: buyer approval ----------
        if status == "CREATED":
            st.markdown("### Step 2 — Buyer approval on PayPal")
            if order.get("approval_url"):
                st.link_button("🅿️ Approve on PayPal (opens PayPal)", order["approval_url"], type="primary")
            st.caption("After approving, PayPal returns you to this dashboard automatically. "
                       "If you approved in another tab, click below to re-check with PayPal.")
            if st.button("🔄 I approved — verify with PayPal"):
                code, body = api_call("POST", f"/payments/{order_id}/verify-approval")
                if code != 200:
                    st.session_state["last_action_error"] = f"Not approved yet ({code}): {server_message(body)}"
                st.rerun()

        elif status == "CANCELLED":
            st.warning("The buyer cancelled this order on PayPal. It can never be captured.")

        # ---------- STEP 3: pre-capture risk gate ----------
        elif status == "APPROVED":
            src = order.get("approval_source")
            st.success(f"Buyer approval verified ({'with PayPal API' if src == 'PAYPAL_API' else 'simulated'}).")
            st.markdown("### Step 3 — Pre-capture risk gate")
            if st.button("⚡ Run pre-capture risk analysis", type="primary"):
                payload = {
                    "transaction_id": f"TX-{order_id}",
                    "order_id": order_id,
                    "amount": order["amount"], "currency": order["currency"],
                    "sender_id": ctx.get("sender_id") or "unknown_sender",
                    "recipient_id": ctx.get("recipient_id") or "unknown_recipient",
                    "recipient_wallet": ctx.get("recipient_wallet"),
                    "device_id": ctx.get("device_id"),
                    "sender_country": ctx.get("sender_country") or "US",
                    "recipient_country": ctx.get("recipient_country") or "US",
                }
                code, body = api_call("POST", "/fraud/analyze", json=payload)
                if code != 200:
                    st.session_state["last_action_error"] = f"Risk analysis failed ({code}): {server_message(body)}"
                st.rerun()

        # ---------- STEP 4: decision + capture ----------
        else:
            tx_code, tx = api_call("GET", f"/fraud/score/{order.get('transaction_id')}")
            st.markdown("### Step 3 — Risk decision")
            if tx_code == 200:
                render_risk_result(tx)
            else:
                st.warning("Stored risk evaluation could not be loaded.")

            st.markdown("### Step 4 — Capture")
            action = order.get("risk_action")
            if status == "CAPTURED" and order.get("mock_mode"):
                st.success("✅ **Captured (MOCK — simulated, no PayPal call).** "
                           "Configure Sandbox credentials for a real capture.")
            elif status == "CAPTURED":
                st.success(f"✅ **Captured by PayPal Sandbox.** Capture ID `{order.get('capture_id')}` · "
                           f"PayPal status `{order.get('paypal_status')}`")
            elif status == "CAPTURE_IN_PROGRESS":
                st.info("Capture in progress…")
            else:
                if status == "CAPTURE_FAILED":
                    st.error(f"Previous capture attempt failed: `{order.get('last_error')}`")
                step_up_status = order.get("step_up_status")
                capture_allowed_now = (action == "CAPTURE"
                                       or (action == "STEP_UP" and step_up_status == "PASSED")
                                       or order.get("analyst_decision") == "APPROVE")
                if action == "CAPTURE":
                    st.caption("The server permits capture for LOW-risk orders.")
                elif action == "STEP_UP":
                    render_step_up(order_id, step_up_status)
                elif order.get("analyst_decision") == "REJECT":
                    st.error(f"🛑 Rejected by analyst `{order.get('analyst_id')}`: {md(order.get('analyst_notes'))}. "
                             "Capture is permanently blocked.")
                elif order.get("analyst_decision") == "APPROVE":
                    st.info(f"Approved by analyst `{order.get('analyst_id')}`; the capture attempt did not complete.")
                else:
                    st.error("Held for analyst review (see the Investigation Queue tab). "
                             "The server will refuse capture until an analyst approves.")
                label = "💸 Capture payment" if capture_allowed_now else "Try capture anyway (server will decide)"
                if st.button(label, type="primary" if capture_allowed_now else "secondary"):
                    code, body = api_call("POST", f"/payments/capture/{order_id}")
                    if code != 200:
                        st.session_state["last_action_error"] = f"Capture refused by server ({code}): {server_message(body)}"
                    st.rerun()

# ----------------- TAB 2: WALLET CLUSTER GRAPH + EVIDENCE -----------------
with tab2:
    st.subheader("🕸️ PYUSD Wallet Cluster Graph & Evidence")
    st.caption("All wallets and transfers in the demo clusters are **synthetic** (addresses 0xde5a…, no tx hashes). "
               "Indicators are **computed from the transfers** by the backend; labels decide nothing.")

    clusters = demo_data.get("clusters", {})
    cluster_ids = [c["cluster_id"] for c in clusters.values()]
    default_cluster = ("cluster_ring_904" if "Scenario C" in demo_scenario
                       else "cluster_clean_001" if "Scenario A" in demo_scenario else cluster_ids[0] if cluster_ids else None)
    if not cluster_ids:
        st.info("No wallet data found in data/demo_wallets.json.")
    else:
        descriptions = {c["cluster_id"]: c.get("description", c["cluster_id"]) for c in clusters.values()}
        cluster_choice = st.selectbox("Inspect cluster:", cluster_ids,
                                      index=cluster_ids.index(default_cluster) if default_cluster in cluster_ids else 0,
                                      format_func=lambda cid: f"{cid} — {descriptions[cid]}")
        code, report = api_call("GET", f"/wallet-clusters/{cluster_choice}")
        if code != 200:
            st.error(f"Could not load cluster ({code}): {server_message(report)}")
        else:
            evidence_wallets = {w for ind in report["indicators"] for t in ind["evidence"]["transfers"]
                                for w in (t["from"], t["to"])}
            nodes = [{"address": m["address"], "label": m.get("label"), "provenance": m.get("provenance"),
                      "allowlisted_merchant": m.get("allowlisted_merchant", False),
                      "is_flagged": m["address"] in evidence_wallets} for m in report["members"]]
            html_graph = generate_interactive_graph_html(nodes=nodes, edges=report["transfers"],
                                                         target_address=report["target_wallet"], height="460px")
            # The HTML is generated locally from validated addresses and escaped labels (see graph_viz.py).
            st.iframe(html_graph, height=480)

            c1, c2, c3 = st.columns(3)
            c1.metric("Wallets", len(report["members"]))
            c2.metric("Transfers", len(report["transfers"]))
            c3.metric("Indicators found", len(report["indicators"]))

            if not report["indicators"]:
                st.success("No suspicious-pattern indicator was found in this cluster's transfers.")
            for ind in report["indicators"]:
                ev = ind["evidence"]
                st.error(f"**{ind['pattern_type']}** — {md(ind['details'])} · "
                         f"Time window: {ev['time_window_start']} → {ev['time_window_end']} "
                         f"({ev['window_hours']} h) · data: {', '.join(ev['provenance'])}")
                st.dataframe([{"transfer": t["transfer_id"], "time (UTC)": t["timestamp"],
                               "from": t["from"], "to": t["to"], "PYUSD": t["amount_pyusd"],
                               "tx hash": t["tx_hash"] or "— (synthetic)", "data": t["provenance"]}
                              for t in ev["transfers"]], hide_index=True, width="stretch")
            st.info("💡 " + report["disclaimer"])

    real_path = Path(__file__).resolve().parent.parent / "data" / "real_pyusd_transfers.json"
    with st.expander("🌐 Real PYUSD data ingested from Ethereum mainnet", expanded=False):
        if not real_path.exists():
            st.caption("No real data fetched yet. Run: `.\\.venv\\Scripts\\python.exe scripts\\fetch_pyusd_transfers.py --limit 100`")
        else:
            real = json.loads(real_path.read_text(encoding="utf-8"))
            meta, rows = real.get("_metadata", {}), real.get("transfers", [])
            st.markdown(f"**{meta.get('transfer_count', len(rows))} real transfers** · {meta.get('network')} · "
                        f"source: {meta.get('source')} · fetched {str(meta.get('fetched_at', ''))[:19]} UTC · "
                        f"{meta.get('rows_dropped_failed_contract_check', 0)} rows dropped by the official-contract check")
            st.caption(f"Token contract (verified official PYUSD): `{meta.get('token_contract')}`. "
                       "Real addresses are pseudonymous and shown without labels or risk judgements; "
                       "they are used to demonstrate verifiable ingestion, not to accuse anyone.")
            st.dataframe([{"time (UTC)": r["timestamp"], "PYUSD": r["amount_pyusd"],
                           "from": r["from"], "to": r["to"],
                           "tx": f"https://etherscan.io/tx/{r['tx_hash']}"} for r in rows[:15]],
                         hide_index=True, width="stretch",
                         column_config={"tx": st.column_config.LinkColumn("tx (Etherscan)", display_text="view")})

def render_case_report(rep):
    """AI case report: provenance of the text, grounded red flags, advisory-only recommendation."""
    if not rep:
        st.info("🤖 Case report is being generated in the background (refresh in a few seconds).")
        return
    if rep.get("llm_status") == "FAILED":
        st.error(f"Case report generation failed ({rep.get('error')}). Review the stored evidence directly.")
        return
    if rep.get("llm_status") == "GEMINI":
        st.caption(f"🤖 Written by **{rep.get('generated_by')}** from the numbered evidence below; "
                   "every red flag cites its evidence and was validated before display.")
    else:
        st.caption(f"📄 **{rep.get('generated_by', 'Deterministic summary')}** — Gemini not used: "
                   f"{rep.get('fallback_reason') or 'n/a'}.")
    st.markdown("### 📌 Executive Summary")
    st.markdown(md(rep.get("executive_summary", "")))
    st.markdown("### 🚨 Top Red Flags")
    for flag in rep.get("top_red_flags", []):
        if isinstance(flag, dict):
            st.markdown(f"* ⚠️ {md(flag.get('text'))} `[{', '.join(flag.get('evidence_ids', []))}]`")
        else:  # reports created before Phase 5
            st.markdown(f"* ⚠️ {flag}")
    st.markdown("### 🎯 Suggested Analyst Action (advisory)")
    label = rep.get("recommended_action_label") or rep.get("recommended_action", "")
    st.warning(f"**{label}**" + (f" — {md(rep['action_rationale'])}" if rep.get("action_rationale") else ""))
    if rep.get("advisory_notice"):
        st.caption(rep["advisory_notice"])
    if rep.get("evidence"):
        with st.expander(f"Evidence items ({len(rep['evidence'])}) · categories sent to the LLM: "
                         f"{', '.join(rep.get('evidence_categories_sent') or []) or 'none'}"):
            for e in rep["evidence"]:
                st.markdown(f"- **{e['id']}** `{e['category']}` — {md(e['text'])}")


# ----------------- TAB 3: ACTIONABLE AI INVESTIGATION QUEUE (ENHANCEMENT 2) -----------------
with tab3:
    st.subheader("📋 Actionable AI Investigation Queue (Human-in-the-Loop)")
    st.markdown("High-risk holds and failed step-up verifications are routed here for a human decision. "
                "**Approve** runs the normal server-side capture; **Reject** permanently blocks it.")
    st.text_input("Your analyst API key", key="analyst_key", type="password",
                  placeholder="paste your key from .env (ANALYST_API_KEYS)",
                  help="Your identity is taken from this key by the server; it is never typed in.")
    st.caption("🔐 Decisions are recorded under the analyst identity bound to this key (prototype: static keys from .env).")

    try:
        alerts_resp = requests.get(f"{API_BASE}/alerts", headers=MERCHANT_HEADERS, timeout=10.0)
        alerts = alerts_resp.json() if alerts_resp.status_code == 200 else []
    except Exception:
        alerts = []

    if not alerts:
        st.info("No pending alerts in queue. Evaluate **Scenario C** in the simulator to generate a pre-capture hold alert!")
    else:
        for alert in alerts:
            alert_id = alert["alert_id"]
            is_resolved = alert.get("status") == "RESOLVED"
            status_badge = "✅ RESOLVED" if is_resolved else "🚨 PENDING ANALYST ACTION"

            with st.expander(f"⚠️ {alert_id} — ${alert['amount']} {alert['currency']} | Risk Score: {alert['risk_score']}/100 [{status_badge}]", expanded=True):
                st.markdown(f"**Transaction ID:** `{alert['transaction_id']}` | **Recipient:** `{alert['recipient_id']}` | **Status:** `{alert['status']}`")

                llm_rep = alert.get("llm_report")
                render_case_report(llm_rep)

                # Case report PDF: fetched only when asked (each export is audit-logged server-side).
                pdf_key = f"case_pdf_{alert_id}"
                if st.button("📄 Prepare case report (PDF)", key=f"prep_{alert_id}"):
                    analyst_key = st.session_state.get("analyst_key", "").strip()
                    try:
                        resp = requests.get(f"{API_BASE}/alerts/{alert_id}/case-report.pdf", timeout=30.0,
                                            headers={"X-API-Key": analyst_key} if analyst_key else MERCHANT_HEADERS)
                        if resp.status_code == 200:
                            st.session_state[pdf_key] = resp.content
                        else:
                            st.error(f"Could not generate report ({resp.status_code}): {server_message(resp.json())}")
                    except requests.RequestException as exc:
                        st.error(f"Backend unreachable: {exc}")
                if st.session_state.get(pdf_key):
                    st.download_button("⬇️ Download PayGuard AI – Suspicious Activity Case Report",
                                       data=st.session_state[pdf_key], file_name=f"payguard_case_{alert_id}.pdf",
                                       mime="application/pdf", key=f"dl_{alert_id}")
                    st.caption("Internal investigation record from stored evidence — not a regulatory SAR filing.")

                if not is_resolved:
                    st.markdown("#### ⚖️ Analyst Decision")
                    notes = st.text_input("Reason for your decision (required, min. 5 characters)",
                                          key=f"notes_{alert_id}")
                    b1, b2 = st.columns(2)
                    decision = None
                    with b1:
                        if st.button("✅ Approve — capture via PayPal", key=f"app_{alert_id}"):
                            decision = "APPROVE"
                    with b2:
                        if st.button("🛑 Reject — block capture", key=f"rej_{alert_id}", type="primary"):
                            decision = "REJECT"
                    if decision and not st.session_state.get("analyst_key", "").strip():
                        st.error("Enter your analyst API key at the top of this tab first.")
                    elif decision:
                        code, body = api_call("POST", f"/alerts/{alert_id}/review", json={
                            "analyst_decision": decision,
                            "analyst_notes": notes.strip(),
                        }, headers={"X-API-Key": st.session_state.get("analyst_key", "").strip()})
                        if code == 200:
                            st.session_state[f"review_result_{alert_id}"] = body.get("capture", {})
                        else:
                            st.session_state[f"review_error_{alert_id}"] = f"({code}) {server_message(body)}"
                        st.rerun()
                    if st.session_state.get(f"review_error_{alert_id}"):
                        st.error(f"Decision not recorded {st.session_state.pop(f'review_error_{alert_id}')}")
                else:
                    st.markdown(f"**Decision:** `{alert.get('analyst_decision')}` by `{alert.get('analyst_id') or 'unknown'}` "
                                f"at {alert.get('decided_at') or alert.get('updated_at')}")
                    st.markdown(f"**Reason:** {md(alert.get('analyst_notes'))}")
                    cap_status, cap_detail = alert.get("capture_status"), alert.get("capture_detail") or ""
                    if cap_status == "CAPTURED":
                        st.success(f"✅ Captured · {cap_detail}")
                    elif cap_status == "CAPTURE_FAILED":
                        st.error(f"❌ Analyst approved, but capture FAILED: {cap_detail}")
                    elif cap_status == "BLOCKED_BY_ANALYST":
                        st.warning("🛑 Capture blocked by analyst. The payment will not be captured.")
                    elif cap_status == "NO_LINKED_ORDER":
                        st.info("Decision recorded. This alert has no linked PayPal order, so nothing was captured.")

# ----------------- TAB 4: OFFICIAL PAYPAL WEBHOOKS (ENHANCEMENT 3) -----------------
with tab4:
    st.subheader("🔔 Official PayPal Webhook Signature Verification")
    st.markdown("""
    Every incoming PayPal webhook is verified with PayPal's official API
    (`POST /v1/notifications/verify-webhook-signature`). Unverified events are rejected (HTTP 401),
    each real event id is processed only once, and capture events are reconciled with risk-gated captures.
    """)

    w_col1, w_col2 = st.columns([1, 1])

    with w_col1:
        st.markdown("#### 🧪 Forged-webhook test")
        st.caption("Sends an event with a **fake** PayPal signature, as an attacker would. "
                   "The backend should reject it. Real events arrive from PayPal via the registered webhook URL.")
        webhook_event_type = st.selectbox(
            "Select Webhook Event:",
            ["PAYMENT.CAPTURE.COMPLETED", "CHECKOUT.ORDER.APPROVED", "PAYMENT.CAPTURE.DENIED"]
        )

        test_webhook_btn = st.button("🚨 Send forged webhook", type="secondary")

        if test_webhook_btn:
            sample_body = {
                "id": f"WH-EVENT-{int(time.time())}",
                "event_type": webhook_event_type,
                "summary": f"PayPal event: {webhook_event_type}",
                "resource": {
                    "id": f"ORDER-SANDBOX-{int(time.time())}",
                    "status": "APPROVED" if "APPROVED" in webhook_event_type else "DENIED"
                }
            }
            sample_headers = {
                "paypal-auth-algo": "SHA256withRSA",
                "paypal-cert-url": "https://api.sandbox.paypal.com/v1/notifications/certs/CERT-360-XYZ",
                "paypal-transmission-id": f"trans-{int(time.time())}",
                "paypal-transmission-sig": "EYiK...SAMPLE_VALID_SIGNATURE...g==",
                "paypal-transmission-time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "Content-Type": "application/json"
            }

            try:
                wh_resp = requests.post(f"{API_BASE}/webhooks/paypal", json=sample_body, headers=sample_headers, timeout=15.0)
                if wh_resp.status_code == 401:
                    st.success(f"✅ Forged webhook REJECTED (HTTP 401): {wh_resp.json().get('detail')}")
                else:
                    st.error(f"⚠️ Unexpected: forged webhook returned {wh_resp.status_code} — {wh_resp.text}")
            except Exception as e:
                st.error(f"Error dispatching webhook: {e}")

    with w_col2:
        st.markdown("#### 📜 Webhook Audit Feed")
        try:
            feed_resp = requests.get(f"{API_BASE}/webhooks/paypal/events", headers=MERCHANT_HEADERS, timeout=10.0)
            if feed_resp.status_code == 200:
                events = feed_resp.json()
                if events:
                    for ev in events[:8]:  # API returns newest first
                        st.code(f"[{ev.get('event_type')}] {ev.get('event_id')} | {ev.get('status')} "
                                f"| signature: {ev.get('verification_status')}")
                else:
                    st.info("No webhook events logged yet.")
        except Exception:
            st.info("Webhook server offline.")
