"""
Pydantic Schemas for PayGuard AI.
Defines data structures for transactions, risk evaluations, alerts, and demo payloads.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime, timezone

ETH_ADDRESS_PATTERN = r"^0x[0-9a-fA-F]{40}$"
SAFE_ID_PATTERN = r"^[A-Za-z0-9._@\-]{1,64}$"     # sender/recipient/device labels
COUNTRY_PATTERN = r"^[A-Z]{2}$"                    # ISO 3166-1 alpha-2
CURRENCY_PATTERN = r"^[A-Z]{3}$"                   # ISO 4217
EMAIL_PATTERN = r"^[^@\s]{1,64}@[^@\s]{1,189}\.[A-Za-z]{2,24}$"
MAX_AMOUNT = 100_000.0                             # prototype ceiling per payment


class RiskBand(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DecisionAction(str, Enum):
    CAPTURE = "CAPTURE"
    STEP_UP = "STEP_UP"
    HOLD = "HOLD"


class PaymentCreateRequest(BaseModel):
    amount: float = Field(..., gt=0, le=MAX_AMOUNT, description="Payment amount")
    currency: str = Field(default="USD", pattern=CURRENCY_PATTERN, description="ISO 4217 currency code")
    recipient_email: str = Field(..., max_length=254, pattern=EMAIL_PATTERN, description="Recipient email (not stored)")
    recipient_wallet: Optional[str] = Field(None, pattern=ETH_ADDRESS_PATTERN, description="Linked Ethereum / PYUSD wallet address (optional, full 42-char 0x address)")
    device_id: Optional[str] = Field(None, pattern=SAFE_ID_PATTERN, description="Simulated device label")
    sender_country: Optional[str] = Field(default="US", pattern=COUNTRY_PATTERN)
    recipient_country: Optional[str] = Field(default="US", pattern=COUNTRY_PATTERN)
    description: Optional[str] = Field(default="PayPal Transaction", max_length=127)
    # Checkout context stored with the order so the risk analysis can resume after the
    # buyer returns from PayPal (the browser session does not survive the redirect).
    demo_scenario: Optional[str] = Field(None, max_length=80, description="Dashboard demo scenario label")
    sender_id: Optional[str] = Field(None, pattern=SAFE_ID_PATTERN)
    recipient_id: Optional[str] = Field(None, pattern=SAFE_ID_PATTERN)
    # No client-supplied history: the server computes the behavioural baseline itself.


class PaymentResponse(BaseModel):
    order_id: str
    status: str
    approval_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FraudAnalyzeRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    order_id: Optional[str] = Field(
        None, pattern=r"^[A-Za-z0-9\-]{1,64}$",
        description="PayPal order this evaluation gates. Required for the decision to permit capture."
    )
    sender_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    amount: float = Field(..., gt=0, le=MAX_AMOUNT)
    currency: str = Field("USD", pattern=CURRENCY_PATTERN)
    recipient_id: str = Field(..., pattern=SAFE_ID_PATTERN)
    recipient_wallet: Optional[str] = Field(None, pattern=ETH_ADDRESS_PATTERN, description="Full 42-char 0x Ethereum address")
    sender_country: str = Field("US", pattern=COUNTRY_PATTERN)
    recipient_country: str = Field("US", pattern=COUNTRY_PATTERN)
    device_id: Optional[str] = Field(None, pattern=SAFE_ID_PATTERN, description="Simulated device label")
    ip_address: Optional[str] = Field(None, max_length=45)
    # historical_avg_amount / transactions_last_24h were removed: any such client values are
    # ignored, and the baseline is computed server-side from stored history.


class ContributingFactor(BaseModel):
    factor: str
    impact: float
    description: str


class ModelExplanation(BaseModel):
    """One SHAP contribution from the PaySim-trained XGBoost model (log-odds units)."""
    feature: str
    label: str
    value: str
    shap_value: float
    direction: str


class ClusterFinding(BaseModel):
    cluster_id: Optional[str] = None
    pattern_type: Optional[str] = None
    num_wallets: int = 0
    details: str = ""
    # Evidence: exact transfers behind the finding and their time window / provenance.
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    provenance: List[str] = Field(default_factory=list)
    evidence_transfers: List[Dict[str, Any]] = Field(default_factory=list)


class FraudAnalyzeResponse(BaseModel):
    transaction_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_band: RiskBand
    action: DecisionAction
    decision_latency_ms: float
    reasons: List[str]
    contributing_factors: List[ContributingFactor]          # rule + wallet points actually added
    model_explanations: List[ModelExplanation] = Field(default_factory=list)  # SHAP top 3
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    cluster_findings: Optional[ClusterFinding] = None
    requires_analyst_review: bool
    signal_availability: Dict[str, str] = Field(default_factory=dict)
    policy_overrides: List[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StepUpVerifyRequest(BaseModel):
    code: str = Field(..., pattern=r"^\d{6}$", description="6-digit simulated OTP")


class AlertReviewRequest(BaseModel):
    analyst_decision: str = Field(..., pattern="^(APPROVE|REJECT)$")
    # The analyst's identity comes from their API key (X-API-Key), never from the request body.
    analyst_notes: str = Field(..., min_length=5, max_length=1000, description="Reason for the decision")


class InvestigationReport(BaseModel):
    transaction_id: str
    executive_summary: str
    top_red_flags: List[str]
    blockchain_evidence: Dict[str, Any]
    recommended_action: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
