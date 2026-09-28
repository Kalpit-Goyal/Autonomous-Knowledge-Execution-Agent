"""Contracts exchanged between the graph, the tools and the API.

These models are the contract with the LLM: the plan, the reconciliation
verdict and the decision set are produced through ``with_structured_output``
against these schemas, so every reasoning step has a typed, persistable shape.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# knowledge sources
# --------------------------------------------------------------------------- #


class SourceType(str, Enum):
    KB = "kb"
    OPS_DB = "ops_db"
    POLICY = "policy"
    CATALOG = "catalog"


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    source_type: SourceType
    source_name: str
    citation: str
    snippet: str
    score: float = 0.0
    query: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


class RetrievalTask(BaseModel):
    step_id: str
    source_type: SourceType
    query: str
    purpose: str
    sql: str = Field(
        default="",
        description=(
            "Only for source_type ops_db: a single SELECT against the operations schema. "
            "Left empty for the semantic and structured sources."
        ),
    )
    tool: str = Field(
        default="",
        description="Read tool to call. Derived from source_type when the planner leaves it blank.",
    )


class PlanStep(BaseModel):
    id: str
    goal: str
    why: str
    sources: list[SourceType] = Field(default_factory=list)
    tasks: list[RetrievalTask] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #


class ReconcileStatus(str, Enum):
    PROCEED = "proceed"
    RE_RETRIEVE = "re_retrieve"
    ASK_USER = "ask_user"


class Reconciliation(BaseModel):
    status: ReconcileStatus
    reason: str
    conflicts: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)
    user_question: str | None = None


# --------------------------------------------------------------------------- #
# decisions and actions
# --------------------------------------------------------------------------- #


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Decision(BaseModel):
    id: str = Field(
        default="",
        description="Short slug other decisions can refer to in depends_on, e.g. 'credit'.",
    )
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str
    risk: RiskLevel = RiskLevel.LOW
    evidence_refs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class DecisionSet(BaseModel):
    decisions: list[Decision] = Field(default_factory=list)
    answer_without_action: str | None = None
    insufficient_information: str | None = None


class ActionOutcome(BaseModel):
    id: str = Field(default_factory=lambda: new_id("act"))
    action: str
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: int = 0
    executed_by: Literal["agent", "human", "skipped"] = "agent"
    approval_id: str | None = None
    why: str = ""


class ApprovalDecision(BaseModel):
    approval_id: str
    action: str
    approved: bool
    approver: str = "unknown"
    reason: str = ""
    decided_at: str = Field(default_factory=utcnow_iso)


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: new_id("apr"))
    session_id: str
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str
    risk: RiskLevel
    reason_for_approval: str
    evidence_refs: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow_iso)
    status: Literal["pending", "approved", "rejected", "auto_approved"] = "pending"
    decided_by: str | None = None
    decision_reason: str | None = None
    decided_at: str | None = None


# --------------------------------------------------------------------------- #
# trace / verification / memory
# --------------------------------------------------------------------------- #


class TraceStep(BaseModel):
    node: str
    label: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=utcnow_iso)


class Verification(BaseModel):
    achieved: bool
    confidence: float = 0.0
    reasoning: str
    missing: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    key: str
    namespace: tuple[str, ...] = ("agent", "long_term", "facts")
    value: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)


class Intent(str, Enum):
    QUESTION = "question"
    TASK = "task"
    COMPLAINT = "complaint"
    ESCALATION = "escalation_request"
    CHURN_RISK = "churn_risk"
    MIXED = "mixed"


class Intake(BaseModel):
    """What the requester actually asked for, pulled out of free text."""

    intent: Intent = Intent.QUESTION
    summary: str = Field(description="One sentence restatement of the request.")
    customer_id: str = Field(default="", description="C-1001 style id, empty if not named.")
    subscription_id: str = Field(default="", description="S-9001 style id, empty if not named.")
    ticket_id: str = Field(default="", description="T-5001 style id, empty if not named.")
    priority: str = Field(default="", description="P1..P4 when the urgency is stated or obvious.")
    wants_state_change: bool = Field(
        default=False, description="True when the requester wants something changed."
    )
    needs_retrieval: bool = Field(
        default=True, description="False only when there is nothing to look up."
    )
    open_points: list[str] = Field(
        default_factory=list, description="Things the message leaves ambiguous."
    )


# --------------------------------------------------------------------------- #
# API payloads
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str = "anonymous"
    message: str
    auto_approve: bool | None = Field(
        default=None,
        description=(
            "Override the AUTO_APPROVE_IRREVERSIBLE setting for this turn. Intended for demos "
            "and automated tests only; a real deployment should leave it null."
        ),
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    status: Literal["completed", "awaiting_approval", "needs_input", "error"]
    citations: list[Evidence] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Disagreements found between sources, with the governing value.",
    )
    decisions: list[Decision] = Field(default_factory=list)
    actions: list[ActionOutcome] = Field(default_factory=list)
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    plan: Plan | None = None
    intake: Intake | None = None
    verification: Verification | None = None
    user_question: str | None = None
    iterations: int = 0
    elapsed_ms: int = 0
    error: str | None = None


class ApproveRequest(BaseModel):
    approved: bool = Field(
        default=True,
        description="Applies to every request when no per-action list is given.",
    )
    approver: str = "support-lead"
    role: str = Field(
        default="",
        description=(
            "Approver role, checked against policies.approvals.approver_roles for actions "
            "above a policy ceiling. Empty means the role is unverified."
        ),
    )
    reason: str = ""


class SessionSummary(BaseModel):
    session_id: str
    status: str
    created_at: str
    updated_at: str
    turns: int
    summary: str = ""
    pending_approvals: int = 0
