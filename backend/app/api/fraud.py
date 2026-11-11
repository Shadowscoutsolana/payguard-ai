"""
Fraud & Risk Intelligence API Router.
Coordinates Layer 1 (Transaction), Layer 2 (Behavioral), Layer 3 (Blockchain),
Layer 4 (Risk Engine), and Layer 5 (AI Investigation Copilot).
Backed by SQLite persistent storage.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
from typing import Dict, Any, List, Optional
import re
import uuid
from datetime import datetime, timedelta, timezone

from ..schemas import (
    FraudAnalyzeRequest,
    FraudAnalyzeResponse,
    AlertReviewRequest,
    InvestigationReport,
    ETH_ADDRESS_PATTERN,
)
from ..transaction_intelligence.feature_extractor import TransactionFeatureExtractor
from ..behavioral_detection.anomaly_detector import BehavioralAnomalyDetector
from ..blockchain_intelligence.wallet_profile import WalletProfileService
from ..blockchain_intelligence.graph import BlockchainGraphEngine
from ..blockchain_intelligence.cluster_detection import ClusterDetector
from ..risk_engine.scoring import RiskEngine
from ..investigation_agent.anonymizer import build_case_evidence
from ..investigation_agent.case_report import gather_case, render_case_pdf
from ..investigation_agent.task_manager import task_manager
from ..database import db
from . import payments
from ..core.security import require_analyst, require_merchant, require_merchant_or_analyst

router = APIRouter(tags=["Fraud & Risk Intelligence"])

# Initialize singletons for low-latency in-memory execution
tx_extractor = TransactionFeatureExtractor()
behavior_detector = BehavioralAnomalyDetector()
wallet_service = WalletProfileService()
graph_engine = BlockchainGraphEngine()
cluster_detector = ClusterDetector(graph_engine, wallet_service)
risk_engine = RiskEngine()


@router.post("/fraud/analyze", response_model=FraudAnalyzeResponse, dependencies=[Depends(require_merchant)])
async def analyze_transaction(req: FraudAnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Synchronous Pre-Capture Risk Gate.
    Target SLA: < 100ms.
    Combines transaction features, behavioral anomaly baselines, and PYUSD wallet graph clustering.
    Persists evaluation to SQLite database.
    """
    payload = req.model_dump()

    # Transaction records are audit evidence: never overwrite an existing one.
    if db.get_transaction(req.transaction_id):
        raise HTTPException(status_code=409, detail="transaction_id already exists; use a new id.")

    # If this evaluation gates a PayPal order, validate the link before scoring.
    order = None
    if req.order_id:
        order = db.get_payment_order(req.order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Unknown order_id.")
        if order["status"] == "CREATED":
            raise HTTPException(status_code=409, detail="Buyer has not approved this order yet; verify approval first.")
        if order["status"] != "APPROVED" or order["transaction_id"]:
            raise HTTPException(status_code=409, detail="This order already has a risk decision and cannot be re-scored.")
        # Score the order's real amount, not whatever the client claims.
        if round(req.amount, 2) != round(order["amount"], 2) or req.currency != order["currency"]:
            raise HTTPException(status_code=400, detail="Amount/currency do not match the PayPal order.")

    # 0. Server-side behavioural baseline (stored captured payments + labelled demo seed).
    #    Counted BEFORE this transaction is saved, so it never includes itself.
    since_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    profile = behavior_detector.build_profile(
        req.sender_id,
        completed_payments=db.get_completed_payments(req.sender_id),
        attempts_last_24h=db.count_attempts_since(req.sender_id, since_24h),
        recipient_id=req.recipient_id,
        stored_recipient_receipts=db.count_completed_receipts(req.recipient_id),
    )

    # 1. Transaction Features (ML domain)
    tx_features = tx_extractor.extract_features(payload, profile)

    # 2. Behavioral Features (ML domain)
    behavior_features = behavior_detector.detect_anomalies(payload, profile)

    # 3. Blockchain Cluster Intelligence (Rule-based domain)
    cluster_analysis = cluster_detector.analyze_cluster(req.recipient_wallet)

    # 4. Multi-Model Risk Evaluation (Separates ML features from Blockchain rule gate)
    evaluation = risk_engine.evaluate(tx_features, behavior_features, cluster_analysis)

    # Bind the decision to the order exactly once (atomic; loses cleanly on a race).
    if order and not db.attach_risk_decision(
        order_id=req.order_id,
        transaction_id=req.transaction_id,
        risk_score=evaluation["risk_score"],
        risk_band=evaluation["risk_band"].value,
        risk_action=evaluation["action"].value,
    ):
        raise HTTPException(status_code=409, detail="This order already has a risk decision and cannot be re-scored.")

    # Record to SQLite database
    db_record = {
        "transaction_id": req.transaction_id,
        "sender_id": req.sender_id,
        "recipient_id": req.recipient_id,
        "amount": req.amount,
        "currency": req.currency,
        "recipient_wallet": req.recipient_wallet,
        "device_id": req.device_id,
        "risk_score": evaluation["risk_score"],
        "risk_band": evaluation["risk_band"].value,
        "action": evaluation["action"].value,
        "decision_latency_ms": evaluation["decision_latency_ms"],
        "reasons": evaluation["reasons"],
        "contributing_factors": [f.model_dump() for f in evaluation["contributing_factors"]],
        "signal_availability": evaluation["signal_availability"],
        "policy_overrides": evaluation["policy_overrides"],
        "model_explanations": evaluation["model_explanations"],
        "score_breakdown": evaluation["score_breakdown"],
        "raw_payload": {
            **payload,
            # What the server actually used (counts only; the recipient/device lists are not copied).
            "server_computed_profile": {
                "baseline_status": profile["baseline_status"],
                "history_count": profile["history_count"],
                "seed_count": profile["seed_count"],
                "stored_count": profile["stored_count"],
                "avg_amount": profile["avg_amount"],
                "attempts_last_24h": profile["attempts_last_24h"],
                "recipient_prior_receipts": profile["recipient_prior_receipts"],
                "device_seen_before": not behavior_features["is_unknown_device"],
                "recipient_seen_before": not behavior_features["is_new_recipient"],
            },
        }
    }
    db.save_transaction(db_record)

    # If HIGH RISK, queue for human-in-the-loop analyst review and async LLM briefing
    if evaluation["requires_analyst_review"]:
        alert_id = f"ALERT-{req.transaction_id}"
        alert_data = {
            "alert_id": alert_id,
            "transaction_id": req.transaction_id,
            "status": "PENDING_REVIEW",
            "risk_score": evaluation["risk_score"],
            "risk_band": evaluation["risk_band"].value,
            "amount": req.amount,
            "currency": req.currency,
            "recipient_id": req.recipient_id,
            "recipient_wallet": req.recipient_wallet,
            "cluster_findings": evaluation.get("cluster_findings").model_dump() if evaluation.get("cluster_findings") else None,
            "reasons": evaluation["reasons"],
            "llm_report": None
        }
        db.save_alert(alert_data)

        # AI case report from STORED evidence only (minimised; no IDs or addresses), in the background.
        pack = build_case_evidence(db.get_transaction(req.transaction_id), alert_data["cluster_findings"])
        background_tasks.add_task(task_manager.execute_investigation, alert_id, pack)

    return FraudAnalyzeResponse(
        transaction_id=req.transaction_id,
        risk_score=evaluation["risk_score"],
        risk_band=evaluation["risk_band"],
        action=evaluation["action"],
        decision_latency_ms=evaluation["decision_latency_ms"],
        reasons=evaluation["reasons"],
        contributing_factors=evaluation["contributing_factors"],
        model_explanations=evaluation["model_explanations"],
        score_breakdown=evaluation["score_breakdown"],
        cluster_findings=evaluation["cluster_findings"],
        requires_analyst_review=evaluation["requires_analyst_review"],
        signal_availability=evaluation["signal_availability"],
        policy_overrides=evaluation["policy_overrides"],
    )


@router.get("/fraud/score/{transaction_id}", dependencies=[Depends(require_merchant_or_analyst)])
async def get_transaction_score(transaction_id: str):
    """Retrieve audit history and score for a transaction from SQLite database."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", transaction_id):
        raise HTTPException(status_code=400, detail="Invalid transaction id.")
    record = db.get_transaction(transaction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Transaction not found in database.")
    return record


def _validate_wallet_path(chain: str, address: str) -> None:
    if chain.lower() != "ethereum":
        raise HTTPException(status_code=400, detail="Only chain 'ethereum' is supported in this prototype.")
    if not re.fullmatch(ETH_ADDRESS_PATTERN, address):
        raise HTTPException(status_code=400, detail="Address must be a full 42-character 0x Ethereum address.")


@router.get("/wallet-clusters/{cluster_id}")
async def get_wallet_cluster(cluster_id: str):
    """Cluster members (with provenance), all transfers, and computed indicators with evidence."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", cluster_id):
        raise HTTPException(status_code=400, detail="Invalid cluster id.")
    report = cluster_detector.cluster_report(cluster_id)
    if not report:
        raise HTTPException(status_code=404, detail="Unknown cluster id.")
    return report


@router.get("/wallets/{chain}/{address}/risk")
async def get_wallet_risk(chain: str, address: str):
    """Look up cached wallet risk and cluster membership."""
    _validate_wallet_path(chain, address)
    profile = wallet_service.get_wallet_profile(address)
    metrics = graph_engine.get_node_metrics(address)
    cluster = cluster_detector.analyze_cluster(address)
    return {
        "chain": chain,
        "address": address,
        "profile": profile,
        "metrics": metrics,
        "cluster_analysis": cluster
    }


@router.get("/wallets/{chain}/{address}/graph")
async def get_wallet_graph(chain: str, address: str, depth: int = 2):
    """Export wallet network subgraph for visual rendering."""
    _validate_wallet_path(chain, address)
    subgraph = graph_engine.export_subgraph(address, depth=max(1, min(depth, 3)))
    return {
        "chain": chain,
        "root_address": address,
        "graph": subgraph
    }


@router.get("/alerts", dependencies=[Depends(require_merchant_or_analyst)])
async def list_alerts(status: Optional[str] = Query(None, pattern="^(PENDING_REVIEW|RESOLVED)$")):
    """Returns queued fraud alerts from SQLite database."""
    return db.list_alerts(status=status)


@router.get("/alerts/{alert_id}/case-report.pdf")
async def download_case_report(alert_id: str, principal: dict = Depends(require_merchant_or_analyst)):
    """PayGuard AI - Suspicious Activity Case Report (internal record, not a regulatory SAR)."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,80}", alert_id):
        raise HTTPException(status_code=400, detail="Invalid alert id.")
    case = gather_case(db, alert_id)
    if not case:
        raise HTTPException(status_code=404, detail="Alert not found.")
    exported_by = f"{principal['role']}:{principal['id']}"
    pdf_bytes = render_case_pdf(case, exported_by=exported_by)
    db.log_audit("ALERT", alert_id, "CASE_REPORT_EXPORTED", {"by": exported_by, "bytes": len(pdf_bytes)})
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="payguard_case_{alert_id}.pdf"'})


def _order_eligible_for_analyst(order: Dict[str, Any]) -> bool:
    return (order.get("analyst_decision") is None
            and order.get("status") in ("RISK_EVALUATED", "CAPTURE_FAILED")
            and (order.get("risk_action") == "HOLD"
                 or (order.get("risk_action") == "STEP_UP" and order.get("step_up_status") == "FAILED")))


@router.post("/alerts/{alert_id}/review")
async def review_alert(alert_id: str, review: AlertReviewRequest, analyst: dict = Depends(require_analyst)):
    """
    Analyst decision on a held / escalated payment.
    APPROVE -> runs the normal server-side capture path (same gate, same PayPal call).
    REJECT  -> the decision is stored on the order, so the gate keeps refusing capture.
    The alert is marked captured only when PayPal confirms the capture.
    """
    alert = db.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    if alert.get("status") != "PENDING_REVIEW":
        raise HTTPException(status_code=409, detail=f"Alert already decided ({alert.get('analyst_decision')}).")

    order = db.get_payment_order_by_transaction(alert["transaction_id"])
    if order and not _order_eligible_for_analyst(order):
        raise HTTPException(status_code=409, detail=f"Linked order is {order['status']} and cannot take an analyst decision.")

    # Atomic: only one analyst decision can ever be recorded for this alert.
    updated = db.resolve_alert(alert_id, review.analyst_decision, review.analyst_notes, analyst["id"])
    if not updated:
        raise HTTPException(status_code=409, detail="Alert was decided by someone else.")

    if not order:
        db.set_alert_capture_status(alert_id, "NO_LINKED_ORDER", "Alert is not linked to a PayPal order.")
        return {**db.get_alert(alert_id), "capture": {"status": "NO_LINKED_ORDER"}}

    db.record_order_analyst_decision(order["order_id"], review.analyst_decision,
                                     analyst["id"], review.analyst_notes)

    if review.analyst_decision == "REJECT":
        db.set_alert_capture_status(alert_id, "BLOCKED_BY_ANALYST", "Capture permanently refused by analyst.")
        capture = {"status": "BLOCKED_BY_ANALYST", "order_id": order["order_id"]}
    else:
        try:
            result = await payments.execute_capture(order["order_id"], initiated_by=f"ANALYST:{analyst['id']}")
            mode = "MOCK (simulated, no PayPal call)" if result.get("mock_mode") else "PayPal Sandbox"
            db.set_alert_capture_status(alert_id, "CAPTURED", f"{mode} · capture_id={result.get('capture_id')}")
            capture = {"status": "CAPTURED", "order_id": order["order_id"],
                       "capture_id": result.get("capture_id"), "mock_mode": result.get("mock_mode")}
        except HTTPException as exc:
            db.set_alert_capture_status(alert_id, "CAPTURE_FAILED", str(exc.detail))
            capture = {"status": "CAPTURE_FAILED", "order_id": order["order_id"], "detail": exc.detail}

    return {**db.get_alert(alert_id), "capture": capture}
