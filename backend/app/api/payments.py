"""
Payments API Router.
Handles PayPal order creation and server-side risk-gated capture.

Capture gate rules (enforced here, never trusted from the client):
- The order must exist in our database and have exactly one stored risk decision.
- LOW  (action CAPTURE): capture allowed.
- MEDIUM (action STEP_UP): capture allowed only after step-up verification PASSED.
- HIGH (action HOLD): capture allowed only after an analyst APPROVE decision.
- An order can only be captured once (atomic claim in the database).
"""

import re
import hmac
import uuid
import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from ..config import settings
from ..schemas import PaymentCreateRequest, PaymentResponse, StepUpVerifyRequest
from ..paypal_integration.client import PayPalClient
from ..database import db
from ..core.security import require_merchant, require_merchant_or_analyst
from ..investigation_agent.anonymizer import build_case_evidence
from ..investigation_agent.task_manager import task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["Payments"])
paypal_client = PayPalClient()

# PayPal order ids are uppercase alphanumerics; mock ids add dashes. Anything else is rejected
# before it can be placed into a PayPal API URL path.
ORDER_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-]{1,64}$")


def _validate_order_id(order_id: str) -> str:
    if not ORDER_ID_PATTERN.match(order_id or ""):
        raise HTTPException(status_code=400, detail="Invalid order id format.")
    return order_id


@router.post("/create", response_model=PaymentResponse, dependencies=[Depends(require_merchant)])
async def create_payment(req: PaymentCreateRequest):
    """
    Creates a PayPal order and registers it locally so its risk decision can be linked later.
    """
    try:
        order = await paypal_client.create_order(
            amount=req.amount,
            currency=req.currency,
            description=req.description or "PayGuard Payment"
        )
    except Exception:
        logger.exception("PayPal create_order call failed")
        raise HTTPException(status_code=502, detail="Failed to create PayPal order.")

    order_id = order.get("id")
    if not order_id:
        logger.error("PayPal create_order returned no id: %s", order)
        raise HTTPException(status_code=502, detail="PayPal did not return an order id.")

    approval_url = None
    for link in order.get("links", []):
        # 'payer-action' when payment_source.experience_context is used; 'approve' in older responses.
        if link.get("rel") in ("payer-action", "approve"):
            approval_url = link.get("href")

    db.create_payment_order(
        order_id=order_id,
        amount=round(req.amount, 2),
        currency=req.currency,
        paypal_status=order.get("status", "CREATED"),
        mock_mode=bool(order.get("mock_mode")),
        approval_url=approval_url,
        # Data minimisation: the email is not needed for scoring, so it is not stored.
        checkout_context=req.model_dump(exclude={"amount", "currency", "description", "recipient_email"}),
    )
    db.log_payment_event(
        event_id=f"EVT-CREATE-{order_id}",
        order_id=order_id,
        event_type="PAYMENT.ORDER.CREATED",
        status=order.get("status", "CREATED"),
        details=order
    )

    return PaymentResponse(
        order_id=order_id,
        status=order.get("status", "CREATED"),
        approval_url=approval_url
    )


async def verify_buyer_approval(order_id: str) -> Dict[str, Any]:
    """
    Confirms buyer approval and records it.
    Real mode: asks PayPal for the order and requires status APPROVED plus a matching amount/currency.
    Mock mode: records a clearly labelled SIMULATED approval.
    Never trusts the browser's query string on its own.
    """
    _validate_order_id(order_id)
    order = db.get_payment_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Unknown order.")
    if order["status"] != "CREATED":
        if order.get("approved_at"):
            return {"order_id": order_id, "status": order["status"], "already_approved": True,
                    "approval_source": order.get("approval_source")}
        raise HTTPException(status_code=409, detail=f"Order is {order['status']} and cannot be approved.")

    if order["mock_mode"]:
        payer_id, source = "SIMULATED-PAYER", "SIMULATED"
    else:
        try:
            pp = await paypal_client.get_order(order_id)
        except Exception:
            logger.exception("PayPal get_order failed for %s", order_id)
            raise HTTPException(status_code=502, detail="Could not confirm the order with PayPal.")

        if pp.get("_http_status") != 200 or pp.get("id") != order_id:
            raise HTTPException(status_code=502, detail="PayPal did not return this order.")
        if pp.get("status") != "APPROVED":
            raise HTTPException(status_code=409,
                                detail=f"PayPal reports order status {pp.get('status')}, not APPROVED.")
        try:
            pp_amount = pp["purchase_units"][0]["amount"]
            amount_ok = (round(float(pp_amount["value"]), 2) == round(order["amount"], 2)
                         and pp_amount["currency_code"] == order["currency"])
        except (KeyError, IndexError, TypeError, ValueError):
            amount_ok = False
        if not amount_ok:
            db.log_audit("PAYMENT_ORDER", order_id, "APPROVAL_AMOUNT_MISMATCH", {"paypal": pp.get("purchase_units")})
            raise HTTPException(status_code=409, detail="PayPal order amount does not match the stored order.")
        payer_id, source = (pp.get("payer") or {}).get("payer_id"), "PAYPAL_API"

    if not db.mark_order_approved(order_id, payer_id=payer_id, approval_source=source):
        raise HTTPException(status_code=409, detail="Order state changed; approval not recorded.")
    return {"order_id": order_id, "status": "APPROVED", "already_approved": False, "approval_source": source}


def _dashboard_redirect(**params) -> RedirectResponse:
    return RedirectResponse(f"{settings.DASHBOARD_URL}/?{urlencode(params)}", status_code=303)


@router.get("/return")
async def paypal_return(token: str = Query(..., description="PayPal order id appended by PayPal")):
    """
    PayPal sends the buyer's browser here after approval.
    The server re-checks the order with PayPal, then redirects the buyer to the dashboard.
    """
    try:
        result = await verify_buyer_approval(token)
        return _dashboard_redirect(order_id=token, approval=result["status"])
    except HTTPException as exc:
        return _dashboard_redirect(order_id=token if ORDER_ID_PATTERN.match(token) else "invalid",
                                   approval="FAILED", reason=str(exc.detail)[:200])


@router.get("/cancel")
async def paypal_cancel(token: str = Query(...)):
    """PayPal sends the buyer here if they cancel; the order can then never be captured."""
    if ORDER_ID_PATTERN.match(token):
        db.mark_order_cancelled(token)
        return _dashboard_redirect(order_id=token, approval="CANCELLED")
    return _dashboard_redirect(order_id="invalid", approval="FAILED", reason="Invalid order id.")


@router.post("/{order_id}/verify-approval", dependencies=[Depends(require_merchant)])
async def verify_approval_endpoint(order_id: str):
    """JSON version of the return check, for the dashboard or API clients."""
    return await verify_buyer_approval(order_id)


@router.get("/{order_id}", dependencies=[Depends(require_merchant_or_analyst)])
async def get_payment_order(order_id: str):
    """Current server-side state of an order (status, risk decision, capture result)."""
    _validate_order_id(order_id)
    order = db.get_payment_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Unknown order.")
    return order


# --- Step-up verification (SIMULATED OTP) -------------------------------------
# A real deployment would deliver the code over SMS/email/authenticator. This prototype
# has no delivery channel, so in DEMO_MODE the code is returned to the dashboard and
# clearly labelled as simulated. Only an HMAC of the code is ever stored.

_STEP_UP_KEY = (settings.STEP_UP_SECRET or secrets.token_hex(32)).encode("utf-8")


def _hash_otp(challenge_id: str, code: str) -> str:
    return hmac.new(_STEP_UP_KEY, f"{challenge_id}:{code}".encode("utf-8"), hashlib.sha256).hexdigest()


def _require_pending_step_up(order_id: str) -> Dict[str, Any]:
    _validate_order_id(order_id)
    order = db.get_payment_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Unknown order.")
    if order.get("risk_action") != "STEP_UP" or order.get("status") not in ("RISK_EVALUATED", "CAPTURE_FAILED"):
        raise HTTPException(status_code=409, detail="This order does not require step-up verification.")
    if order.get("step_up_status") != "PENDING":
        raise HTTPException(status_code=409, detail=f"Step-up already {order.get('step_up_status')}.")
    return order


def _escalate_step_up(order: Dict[str, Any], reason: str, background_tasks: Optional[BackgroundTasks] = None) -> None:
    """Step-up failed: block capture permanently unless an analyst approves (Task 2.3)."""
    if not db.set_step_up_result(order["order_id"], "FAILED", reason):
        return
    tx = db.get_transaction(order["transaction_id"]) or {}
    db.save_alert({
        "alert_id": f"ALERT-{order['transaction_id']}",
        "transaction_id": order["transaction_id"],
        "status": "PENDING_REVIEW",
        "risk_score": order.get("risk_score"),
        "risk_band": order.get("risk_band"),
        "amount": order.get("amount"),
        "currency": order.get("currency"),
        "recipient_id": tx.get("recipient_id"),
        "recipient_wallet": tx.get("recipient_wallet"),
        "reasons": [f"Step-up verification failed: {reason}"] + (tx.get("reasons") or []),
    })
    if background_tasks is not None and tx:
        # AI case report for the analyst, from stored evidence only, outside the request path.
        background_tasks.add_task(task_manager.execute_investigation, f"ALERT-{order['transaction_id']}",
                                  build_case_evidence(tx, None, escalation_reason=reason))


@router.post("/{order_id}/step-up/start", dependencies=[Depends(require_merchant)])
async def start_step_up(order_id: str, background_tasks: BackgroundTasks):
    """Issues a new 6-digit code for a MEDIUM-risk order (supersedes any earlier code)."""
    order = _require_pending_step_up(order_id)
    if db.count_step_up_challenges(order_id) >= settings.STEP_UP_MAX_CHALLENGES:
        _escalate_step_up(order, f"{settings.STEP_UP_MAX_CHALLENGES} codes issued without success", background_tasks)
        # Returned (not raised) so the background AI report task still runs.
        return JSONResponse(status_code=403, background=background_tasks,
                            content={"detail": "Too many verification codes requested; escalated to analyst review."})

    challenge_id = uuid.uuid4().hex
    code = f"{secrets.randbelow(10**6):06d}"
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.STEP_UP_OTP_TTL_SECONDS)
    db.create_step_up_challenge(challenge_id, order_id, _hash_otp(challenge_id, code), expires_at.isoformat())

    response = {
        "order_id": order_id,
        "challenge_id": challenge_id,
        "expires_in_seconds": settings.STEP_UP_OTP_TTL_SECONDS,
        "max_attempts": settings.STEP_UP_MAX_ATTEMPTS,
        "delivery": "SIMULATED",
    }
    if settings.DEMO_MODE:
        response["simulated_code"] = code  # demo only: no real SMS/email channel exists
    return response


@router.post("/{order_id}/step-up/verify", dependencies=[Depends(require_merchant)])
async def verify_step_up(order_id: str, req: StepUpVerifyRequest, background_tasks: BackgroundTasks):
    """Verifies the code server-side. PASSED unlocks capture; repeated failure escalates."""
    order = _require_pending_step_up(order_id)
    challenge = db.get_active_step_up_challenge(order_id)
    if not challenge:
        raise HTTPException(status_code=409, detail="No active verification code. Request a new code.")

    if datetime.now(timezone.utc) >= datetime.fromisoformat(challenge["expires_at"]):
        db.close_step_up_challenge(challenge["challenge_id"], "EXPIRED")
        db.log_audit("PAYMENT_ORDER", order_id, "STEP_UP_CODE_EXPIRED", {"challenge_id": challenge["challenge_id"]})
        raise HTTPException(status_code=410, detail="Verification code expired. Request a new code.")

    attempts = db.record_step_up_attempt(challenge["challenge_id"])
    expected = challenge["code_hash"]
    if hmac.compare_digest(_hash_otp(challenge["challenge_id"], req.code), expected):
        db.close_step_up_challenge(challenge["challenge_id"], "PASSED")
        db.set_step_up_result(order_id, "PASSED", "Simulated OTP verified")
        return {"order_id": order_id, "step_up_status": "PASSED",
                "note": "Simulated verification passed; capture is now permitted."}

    remaining = settings.STEP_UP_MAX_ATTEMPTS - attempts
    db.log_audit("PAYMENT_ORDER", order_id, "STEP_UP_WRONG_CODE",
                 {"challenge_id": challenge["challenge_id"], "attempt": attempts})
    if remaining <= 0:
        db.close_step_up_challenge(challenge["challenge_id"], "FAILED")
        _escalate_step_up(order, f"{settings.STEP_UP_MAX_ATTEMPTS} incorrect codes", background_tasks)
        return JSONResponse(status_code=403, background=background_tasks,
                            content={"detail": "Too many incorrect codes; escalated to analyst review."})
    raise HTTPException(status_code=400, detail=f"Incorrect code. {remaining} attempt(s) left.")


def capture_block_reason(order: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    """
    Pure policy check. Returns (http_status, reason) if capture must be blocked, else None.
    """
    status = order.get("status")
    if status == "CREATED":
        return 409, "Buyer has not approved this order on PayPal yet."
    if status == "CANCELLED":
        return 409, "Buyer cancelled this order."
    if status == "CAPTURED":
        return 409, "Order has already been captured."
    if status == "CAPTURE_IN_PROGRESS":
        return 409, "A capture for this order is already in progress."
    if status not in ("RISK_EVALUATED", "CAPTURE_FAILED") or not order.get("risk_action"):
        return 409, "No risk decision is stored for this order. Run the pre-capture risk analysis first."

    action = order["risk_action"]
    if action == "CAPTURE":
        return None
    if action == "STEP_UP":
        if order.get("step_up_status") == "PASSED":
            return None
        if order.get("step_up_status") == "FAILED":
            if order.get("analyst_decision") == "APPROVE":
                return None
            if order.get("analyst_decision") == "REJECT":
                return 403, "Analyst rejected this payment; capture is permanently blocked."
            return 403, "Step-up verification failed; escalated to analyst review."
        return 403, "Medium-risk order: step-up verification has not been passed."
    if action == "HOLD":
        if order.get("analyst_decision") == "APPROVE":
            return None
        if order.get("analyst_decision") == "REJECT":
            return 403, "Analyst rejected this payment; capture is permanently blocked."
        return 403, "High-risk order is on hold pending analyst approval."
    return 403, f"Unrecognised risk action '{action}'; capture refused."


def _extract_capture_id(result: Dict[str, Any]) -> Optional[str]:
    """Pulls the capture id out of a PayPal Orders v2 capture response."""
    try:
        return result["purchase_units"][0]["payments"]["captures"][0]["id"]
    except (KeyError, IndexError, TypeError):
        return None


@router.post("/capture/{order_id}", dependencies=[Depends(require_merchant)])
async def capture_payment(order_id: str):
    """
    Captures the PayPal order only if the stored server-side risk decision permits it.
    Any risk score or override flag sent by the client is ignored.
    """
    return await execute_capture(order_id, initiated_by="API")


async def execute_capture(order_id: str, initiated_by: str) -> Dict[str, Any]:
    """
    The single capture path used by the API and by analyst approval.
    Raises HTTPException when the gate blocks capture or PayPal does not complete it.
    """
    _validate_order_id(order_id)
    order = db.get_payment_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Unknown order. Orders must be created through /payments/create.")

    blocked = capture_block_reason(order)
    if blocked:
        http_status, reason = blocked
        db.log_payment_event(
            event_id=f"EVT-BLOCKED-{order_id}-{uuid.uuid4().hex[:8]}",
            order_id=order_id,
            event_type="PAYMENT.CAPTURE.BLOCKED_BY_RISK_GATE",
            status=order.get("status"),
            details={"reason": reason, "risk_band": order.get("risk_band"),
                     "risk_action": order.get("risk_action"), "initiated_by": initiated_by}
        )
        raise HTTPException(status_code=http_status, detail=reason)

    # Atomic claim: if two requests race, only one gets past this line.
    if not db.claim_order_for_capture(order_id):
        raise HTTPException(status_code=409, detail="Order is no longer eligible for capture (concurrent request).")

    try:
        capture_result = await paypal_client.capture_order(order_id)
    except Exception as exc:
        logger.exception("PayPal capture call failed for %s", order_id)
        db.finish_capture(order_id, success=False, capture_id=None, paypal_status=None,
                          error=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Capture request to PayPal failed.")

    paypal_status = capture_result.get("status")
    success = paypal_status == "COMPLETED"
    capture_id = _extract_capture_id(capture_result)
    db.finish_capture(
        order_id, success=success, capture_id=capture_id, paypal_status=paypal_status,
        error=None if success else (capture_result.get("name") or "CAPTURE_NOT_COMPLETED"),
    )
    db.log_payment_event(
        event_id=f"EVT-CAPTURE-{order_id}-{uuid.uuid4().hex[:8]}",
        order_id=order_id,
        event_type="PAYMENT.CAPTURE.COMPLETED" if success else "PAYMENT.CAPTURE.FAILED",
        status=paypal_status or "UNKNOWN",
        details={**capture_result, "initiated_by": initiated_by}
    )

    if not success:
        raise HTTPException(status_code=502, detail=f"PayPal did not complete the capture (status: {paypal_status}).")

    return {
        "order_id": order_id,
        "status": "CAPTURED",
        "capture_id": capture_id,
        "mock_mode": bool(capture_result.get("mock_mode")),
        "paypal_response": capture_result
    }
