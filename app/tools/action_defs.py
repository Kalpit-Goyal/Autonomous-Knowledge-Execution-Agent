"""The action catalogue.

Each handler performs a real, observable write: a row in SQLite, a file in
``data/outbox``, or both. Nothing is simulated, and every handler returns a
result the graph records in the audit log and the UI renders.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.knowledge import sql_store
from app.knowledge.policy_store import approval_rule, escalation_targets, get_policy, sla_for
from app.schemas import RiskLevel
from app.tools.action_registry import ACTIONS, ActionContext, ActionSpec, register

_TICKET_RE = re.compile(r"^T-\d{4,}$")
_CUSTOMER_RE = re.compile(r"^C-\d{4,}$")


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat(timespec="seconds")


def _next_ticket_id(settings: Settings) -> str:
    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(CAST(SUBSTR(ticket_id,3) AS INTEGER)), 5000) AS n FROM tickets"
        ).fetchone()
    finally:
        conn.close()
    return f"T-{int(row['n']) + 1}"


def _write_outbox(settings: Settings, filename: str, payload: str) -> str:
    settings.outbox_dir.mkdir(parents=True, exist_ok=True)
    target = settings.outbox_dir / filename
    target.write_text(payload, encoding="utf-8")
    return str(target)


def _log_event(ticket_id: str, kind: str, body: str, actor: str, settings: Settings) -> None:
    sql_store.write(
        "INSERT INTO ticket_events (ticket_id, kind, body, actor, created_at) VALUES (?,?,?,?,?)",
        (ticket_id, kind, body, actor, _iso()),
        settings,
    )


def _ticket_exists(ticket_id: str, settings: Settings) -> bool:
    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute("SELECT 1 FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# argument models
# --------------------------------------------------------------------------- #


class CreateTicketArgs(BaseModel):
    customer_id: str = Field(description="Customer id, e.g. C-1004.")
    subject: str = Field(description="Short ticket subject.")
    priority: str = Field(description="Priority: P1, P2, P3 or P4.")
    category: str = Field(default="general", description="Category, e.g. billing, security.")
    body: str = Field(default="", description="What the ticket is about.")

    @field_validator("priority")
    @classmethod
    def _check_priority(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in {"P1", "P2", "P3", "P4"}:
            raise ValueError("priority must be one of P1, P2, P3, P4")
        return upper

    @field_validator("customer_id")
    @classmethod
    def _check_customer(cls, v: str) -> str:
        if not _CUSTOMER_RE.match(v.strip()):
            raise ValueError("customer_id must look like C-1004")
        return v.strip()


class AddTicketNoteArgs(BaseModel):
    ticket_id: str = Field(description="Ticket id, e.g. T-5001.")
    body: str = Field(description="Note text.")
    visibility: str = Field(
        default="internal", description="'internal' or 'customer_visible'."
    )

    @field_validator("ticket_id")
    @classmethod
    def _check_ticket(cls, v: str) -> str:
        if not _TICKET_RE.match(v.strip()):
            raise ValueError("ticket_id must look like T-5001")
        return v.strip()


class UpdateTicketStatusArgs(BaseModel):
    ticket_id: str = Field(description="Ticket id, e.g. T-5002.")
    status: str = Field(
        description=(
            "New status: open, in_progress, awaiting_customer_response, "
            "awaiting_third_party_provider, resolved, closed."
        )
    )
    reason: str = Field(default="", description="Why the status is changing.")

    @field_validator("ticket_id")
    @classmethod
    def _check_ticket(cls, v: str) -> str:
        if not _TICKET_RE.match(v.strip()):
            raise ValueError("ticket_id must look like T-5001")
        return v.strip()

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        allowed = {
            "open",
            "in_progress",
            "awaiting_customer_response",
            "awaiting_third_party_provider",
            "resolved",
            "closed",
        }
        if v.strip() not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return v.strip()


class EscalateTicketArgs(BaseModel):
    ticket_id: str = Field(description="Ticket id, e.g. T-5001.")
    reason: str = Field(description="Why escalation is warranted.")
    severity: str = Field(
        default="", description="Optional explicit priority, e.g. P1. Defaults to current priority."
    )

    @field_validator("ticket_id")
    @classmethod
    def _check_ticket(cls, v: str) -> str:
        if not _TICKET_RE.match(v.strip()):
            raise ValueError("ticket_id must look like T-5001")
        return v.strip()


class SendCustomerReplyArgs(BaseModel):
    customer_id: str = Field(description="Customer id, e.g. C-1001.")
    subject: str = Field(description="Subject line.")
    body: str = Field(description="Message body sent to the customer.")
    ticket_id: str = Field(default="", description="Related ticket id if any.")

    @field_validator("customer_id")
    @classmethod
    def _check_customer(cls, v: str) -> str:
        if not _CUSTOMER_RE.match(v.strip()):
            raise ValueError("customer_id must look like C-1004")
        return v.strip()


class ScheduleCallbackArgs(BaseModel):
    customer_id: str = Field(description="Customer id, e.g. C-1003.")
    when: str = Field(description="ISO timestamp for the callback.")
    purpose: str = Field(description="What the callback is about.")
    ticket_id: str = Field(default="", description="Related ticket id if any.")

    @field_validator("when")
    @classmethod
    def _check_when(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("when must be an ISO-8601 timestamp") from exc
        return v.strip()


class ExportCaseReportArgs(BaseModel):
    customer_id: str = Field(description="Customer id to report on.")
    format: str = Field(default="csv", description="Output format: csv or json.")
    include: list[str] = Field(
        default_factory=lambda: ["tickets", "subscription"],
        description="Sections to include: tickets, subscription, account.",
    )

    @field_validator("customer_id")
    @classmethod
    def _check_customer(cls, v: str) -> str:
        if not _CUSTOMER_RE.match(v.strip()):
            raise ValueError("customer_id must look like C-1004")
        return v.strip()


class ApplyAccountCreditArgs(BaseModel):
    customer_id: str = Field(description="Customer id to credit.")
    amount_usd: float = Field(gt=0, description="Credit amount in USD.")
    reason: str = Field(description="Justification, tied to a ticket or incident.")
    ticket_id: str = Field(default="", description="Ticket that justifies the credit.")

    @field_validator("customer_id")
    @classmethod
    def _check_customer(cls, v: str) -> str:
        if not _CUSTOMER_RE.match(v.strip()):
            raise ValueError("customer_id must look like C-1004")
        return v.strip()


class CancelSubscriptionArgs(BaseModel):
    subscription_id: str = Field(description="Subscription id, e.g. S-9004.")
    reason: str = Field(description="Why the subscription is being cancelled.")
    at_period_end: bool = Field(
        default=True, description="True cancels at renewal, False cancels immediately."
    )

    @field_validator("subscription_id")
    @classmethod
    def _check_sub(cls, v: str) -> str:
        if not re.match(r"^S-\d{4,}$", v.strip()):
            raise ValueError("subscription_id must look like S-9004")
        return v.strip()


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #


def _create_ticket(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    ticket_id = _next_ticket_id(settings)
    now = _iso()
    sql_store.write(
        "INSERT INTO tickets (ticket_id, customer_id, subject, priority, status, created_at, "
        "updated_at, open_minutes, category, sentiment) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            ticket_id,
            args["customer_id"],
            args["subject"],
            args["priority"],
            "open",
            now,
            now,
            0,
            args.get("category") or "general",
            "neutral",
        ),
        settings,
    )
    _log_event(ticket_id, "created", args.get("body") or args["subject"], ctx.actor, settings)
    return {
        "ticket_id": ticket_id,
        "customer_id": args["customer_id"],
        "priority": args["priority"],
        "status": "open",
        "sla_first_response_minutes": sla_for(args["priority"]).get("first_response_minutes"),
    }


def _add_ticket_note(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    if not _ticket_exists(args["ticket_id"], settings):
        raise ValueError(f"ticket {args['ticket_id']} does not exist")
    _log_event(
        args["ticket_id"],
        f"note:{args.get('visibility', 'internal')}",
        args["body"],
        ctx.actor,
        settings,
    )
    return {"ticket_id": args["ticket_id"], "recorded": True, "visibility": args.get("visibility")}


def _update_ticket_status(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    if not _ticket_exists(args["ticket_id"], settings):
        raise ValueError(f"ticket {args['ticket_id']} does not exist")
    previous = "unknown"
    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute(
            "SELECT status FROM tickets WHERE ticket_id = ?", (args["ticket_id"],)
        ).fetchone()
        previous = row["status"] if row else "unknown"
    finally:
        conn.close()

    sql_store.write(
        "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
        (args["status"], _iso(), args["ticket_id"]),
        settings,
    )
    _log_event(
        args["ticket_id"],
        "status_change",
        f"{previous} -> {args['status']}: {args.get('reason') or 'no reason given'}",
        ctx.actor,
        settings,
    )
    return {"ticket_id": args["ticket_id"], "previous_status": previous, "status": args["status"]}


def _escalate_ticket(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    if not _ticket_exists(args["ticket_id"], settings):
        raise ValueError(f"ticket {args['ticket_id']} does not exist")

    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute(
            "SELECT priority FROM tickets WHERE ticket_id = ?", (args["ticket_id"],)
        ).fetchone()
    finally:
        conn.close()
    current = (row["priority"] if row else "P3") or "P3"
    new_priority = (args.get("severity") or current).upper()
    targets = escalation_targets(new_priority)

    sql_store.write(
        "UPDATE tickets SET priority = ?, status = 'in_progress', "
        "updated_at = ? WHERE ticket_id = ?",
        (new_priority, _iso(), args["ticket_id"]),
        settings,
    )
    _log_event(
        args["ticket_id"],
        "escalation",
        f"{current} -> {new_priority}: {args['reason']}",
        ctx.actor,
        settings,
    )
    notified = list(targets.get("notify", []))
    page = bool(targets.get("page"))
    outbox_file = ""
    if notified:
        filename = f"escalation_{args['ticket_id']}_{int(_now().timestamp())}.md"
        _write_outbox(
            settings,
            filename,
            f"# Escalation {args['ticket_id']}\n\n"
            f"- from priority: {current}\n- to priority: {new_priority}\n"
            f"- reason: {args['reason']}\n- page on-call: {page}\n"
            f"- notified: {', '.join(notified)}\n- raised by: {ctx.actor}\n"
            f"- at: {_iso()}\n\n## Why the agent escalated\n\n{ctx.why}\n",
        )
        outbox_file = filename
    return {
        "ticket_id": args["ticket_id"],
        "previous_priority": current,
        "priority": new_priority,
        "notified": notified,
        "paged": page,
        "sla": sla_for(new_priority),
        "outbox_file": outbox_file,
    }


def _send_customer_reply(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute(
            "SELECT email FROM customers WHERE customer_id = ?", (args["customer_id"],)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"customer {args['customer_id']} not found")

    filename = f"reply_{args['customer_id']}_{int(_now().timestamp())}.md"
    path = _write_outbox(
        settings,
        filename,
        f"To: {row['email']}\nSubject: {args['subject']}\n\n{args['body']}\n\n"
        f"--\nSent by the support agent at {_iso()}.\nRationale: {ctx.why}\n",
    )
    if args.get("ticket_id"):
        _log_event(
            args["ticket_id"], "reply_sent", args["subject"], ctx.actor, settings
        )
    return {
        "delivered_to": row["email"],
        "subject": args["subject"],
        "outbox_file": Path(path).name,
        "ticket_id": args.get("ticket_id") or None,
    }


def _schedule_callback(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    when = datetime.fromisoformat(args["when"].replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    ref = args.get("ticket_id") or args["customer_id"]
    _log_event(
        ref,
        "callback_scheduled",
        json.dumps({"when": _iso(when), "purpose": args["purpose"]}),
        ctx.actor,
        settings,
    )
    return {
        "scheduled_for": _iso(when),
        "in_hours": round((when - _now()).total_seconds() / 3600, 2),
        "reference": ref,
        "purpose": args["purpose"],
    }


def _export_case_report(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    include = set(args.get("include") or [])
    conn = sql_store.get_connection(settings)
    try:
        customer = None
        sections: dict[str, Any] = {}
        if "account" in include or "tickets" in include:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (args["customer_id"],)
            ).fetchone()
            if row is None:
                raise ValueError(f"customer {args['customer_id']} not found")
            customer = dict(row)
            if "account" in include:
                sections["account"] = customer
        if "subscription" in include:
            row = conn.execute(
                "SELECT * FROM subscriptions WHERE customer_id = ?", (args["customer_id"],)
            ).fetchone()
            if row is not None:
                sections["subscription"] = dict(row)
        if "tickets" in include:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE customer_id = ? ORDER BY ticket_id",
                (args["customer_id"],),
            ).fetchall()
            sections["tickets"] = [dict(r) for r in rows]
    finally:
        conn.close()

    fmt = (args.get("format") or "csv").lower()
    if fmt == "json":
        filename = f"case_report_{args['customer_id']}_{int(_now().timestamp())}.json"
        payload = json.dumps(
            {"customer_id": args["customer_id"], "generated_at": _iso(), "sections": sections},
            indent=2,
            default=str,
        )
    else:
        filename = f"case_report_{args['customer_id']}_{int(_now().timestamp())}.csv"
        buffer: list[str] = []
        for section, payload_value in sections.items():
            rows = payload_value if isinstance(payload_value, list) else [payload_value]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                buffer.append(section)
                buffer.append(
                    ",".join(f'"{k}={v}"' for k, v in row.items() if v is not None)
                )
        payload = "section,detail\n" + "\n".join(buffer) + "\n"

    path = _write_outbox(settings, filename, payload)
    return {
        "outbox_file": Path(path).name,
        "format": fmt,
        "sections": sorted(sections),
        "customer": (customer or {}).get("name"),
    }


def _apply_account_credit(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?", (args["customer_id"],)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"customer {args['customer_id']} not found")

    credit_id = f"CR-{int(_now().timestamp())}"
    expires = _iso(_now() + timedelta(days=365))
    sql_store.write(
        "INSERT INTO account_credits (credit_id, customer_id, amount_usd, reason, approved_by, "
        "expires_at, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            credit_id,
            args["customer_id"],
            float(args["amount_usd"]),
            args["reason"],
            ctx.actor,
            expires,
            _iso(),
        ),
        settings,
    )
    if args.get("ticket_id"):
        _log_event(
            args["ticket_id"],
            "credit_issued",
            f"{credit_id} for {args['amount_usd']} USD: {args['reason']}",
            ctx.actor,
            settings,
        )
    return {
        "credit_id": credit_id,
        "amount_usd": args["amount_usd"],
        "approved_by": ctx.actor,
        "approval_id": ctx.approval_id,
        "expires_at": expires,
    }


def _cancel_subscription(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    settings = get_settings()
    conn = sql_store.get_connection(settings)
    try:
        row = conn.execute(
            "SELECT * FROM subscriptions WHERE subscription_id = ?",
            (args["subscription_id"],),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"subscription {args['subscription_id']} not found")

    new_status = "cancelled_at_period_end" if args.get("at_period_end", True) else "cancelled"
    sql_store.write(
        "UPDATE subscriptions SET status = ?, auto_renew = 0 WHERE subscription_id = ?",
        (new_status, args["subscription_id"]),
        settings,
    )
    filename = f"cancellation_{args['subscription_id']}_{int(_now().timestamp())}.json"
    _write_outbox(
        settings,
        filename,
        json.dumps(
            {
                "subscription_id": args["subscription_id"],
                "customer_id": row["customer_id"],
                "previous_status": row["status"],
                "new_status": new_status,
                "at_period_end": bool(args.get("at_period_end", True)),
                "renewal_date": row["renewal_date"],
                "reason": args["reason"],
                "approved_by": ctx.actor,
                "approval_id": ctx.approval_id,
                "rationale": ctx.why,
                "at": _iso(),
            },
            indent=2,
            default=str,
        ),
    )
    return {
        "subscription_id": args["subscription_id"],
        "customer_id": row["customer_id"],
        "previous_status": row["status"],
        "status": new_status,
        "at_period_end": bool(args.get("at_period_end", True)),
        "renewal_date": row["renewal_date"],
        "approved_by": ctx.actor,
    }


class IssuePartialRefundArgs(BaseModel):
    customer_id: str = Field(description="Customer id to refund.")
    amount_usd: float = Field(gt=0, description="Refund amount in USD.")
    reason: str = Field(description="Justification, tied to a ticket or incident.")
    ticket_id: str = Field(default="", description="Ticket that justifies the refund.")
    subscription_id: str = Field(default="", description="Subscription being refunded.")

    @field_validator("customer_id")
    @classmethod
    def _check_customer(cls, v: str) -> str:
        if not _CUSTOMER_RE.match(v.strip()):
            raise ValueError("customer_id must look like C-1004")
        return v.strip()


def _issue_partial_refund(args: dict[str, Any], ctx: ActionContext) -> dict[str, Any]:
    """Issue a refund, enforcing the ceiling and approver role from policy.

    The thresholds are read from ``policies.json`` rather than hardcoded, so
    finance can change them without touching code. Above the ceiling the
    approver must hold one of the roles the policy names.
    """
    settings = get_settings()
    rule = approval_rule()
    ceiling = float(get_policy("refunds").get("partial_refund_ceiling_usd", 0) or 0)
    allowed_roles = [str(r) for r in rule.get("approver_roles", [])]
    amount = float(args["amount_usd"])

    conn = sql_store.get_connection(settings)
    try:
        customer = conn.execute(
            "SELECT customer_id FROM customers WHERE customer_id = ?", (args["customer_id"],)
        ).fetchone()
    finally:
        conn.close()
    if customer is None:
        raise ValueError(f"customer {args['customer_id']} not found")

    over_ceiling = bool(ceiling) and amount > ceiling
    if over_ceiling and allowed_roles:
        role = getattr(ctx, "approver_role", "") or ""
        if role not in allowed_roles:
            raise PermissionError(
                f"refund of {amount} USD exceeds the {ceiling} USD ceiling; it needs approval "
                f"from one of {', '.join(allowed_roles)} (approver role was {role or 'unset'})"
            )

    refund_id = f"RF-{int(_now().timestamp())}"
    sql_store.write(
        "INSERT INTO refunds (refund_id, customer_id, subscription_id, amount_usd, reason, "
        "approved_by, approval_role, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            refund_id,
            args["customer_id"],
            args.get("subscription_id", ""),
            amount,
            args["reason"],
            ctx.actor,
            getattr(ctx, "approver_role", ""),
            _iso(),
        ),
        settings,
    )
    if args.get("ticket_id"):
        _log_event(
            args["ticket_id"],
            "refund_issued",
            f"{refund_id} for {amount} USD: {args['reason']}",
            ctx.actor,
            settings,
        )
    return {
        "refund_id": refund_id,
        "amount_usd": amount,
        "over_ceiling": over_ceiling,
        "ceiling_usd": ceiling,
        "approved_by": ctx.actor,
        "approver_role": getattr(ctx, "approver_role", ""),
        "approval_id": ctx.approval_id,
    }


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #

register(
    ActionSpec(
        name="create_ticket",
        description=(
            "Open a new support ticket for a customer when no existing ticket covers the issue."
        ),
        args_model=CreateTicketArgs,
        fn=_create_ticket,
        risk=RiskLevel.LOW,
        irreversible=False,
        effect="inserts a row in the tickets table",
    )
)

register(
    ActionSpec(
        name="add_ticket_note",
        description="Record an investigation note on an existing ticket.",
        args_model=AddTicketNoteArgs,
        fn=_add_ticket_note,
        risk=RiskLevel.LOW,
        irreversible=False,
        effect="appends a row to ticket_events",
    )
)

register(
    ActionSpec(
        name="update_ticket_status",
        description=(
            "Move a ticket through its lifecycle. Use awaiting_customer_response when the "
            "ball is in the customer's court, since that state stops the SLA clock."
        ),
        args_model=UpdateTicketStatusArgs,
        fn=_update_ticket_status,
        risk=RiskLevel.LOW,
        irreversible=False,
        effect="updates tickets.status and appends a status_change event",
    )
)

register(
    ActionSpec(
        name="escalate_ticket",
        description=(
            "Escalate a ticket to the on-call rotation and the account team, raising its "
            "priority if needed and notifying whoever the policy matrix names for that tier."
        ),
        args_model=EscalateTicketArgs,
        fn=_escalate_ticket,
        risk=RiskLevel.MEDIUM,
        irreversible=False,
        effect="raises priority, sets status in_progress, writes an escalation notice",
    )
)

register(
    ActionSpec(
        name="send_customer_reply",
        description="Send a written response to the customer explaining the position.",
        args_model=SendCustomerReplyArgs,
        fn=_send_customer_reply,
        risk=RiskLevel.LOW,
        irreversible=False,
        effect="writes the message to the outbox and logs it against the ticket",
    )
)

register(
    ActionSpec(
        name="schedule_callback",
        description="Book a callback with the customer at a specific time.",
        args_model=ScheduleCallbackArgs,
        fn=_schedule_callback,
        risk=RiskLevel.LOW,
        irreversible=False,
        effect="records a callback_scheduled event",
    )
)

register(
    ActionSpec(
        name="export_case_report",
        description="Compile a customer's ticket history, account and subscription into a file.",
        args_model=ExportCaseReportArgs,
        fn=_export_case_report,
        risk=RiskLevel.LOW,
        irreversible=False,
        effect="writes a case report file to the outbox",
    )
)

register(
    ActionSpec(
        name="apply_account_credit",
        description=(
            "Apply goodwill credit to a customer's account. This moves money and cannot be "
            "reversed, so it always goes to a human for approval first."
        ),
        args_model=ApplyAccountCreditArgs,
        fn=_apply_account_credit,
        risk=RiskLevel.HIGH,
        irreversible=True,
        effect="inserts a row in account_credits",
    )
)

register(
    ActionSpec(
        name="cancel_subscription",
        description=(
            "Cancel a subscription. This terminates a contract and cannot be reversed, so it "
            "always goes to a human for approval first."
        ),
        args_model=CancelSubscriptionArgs,
        fn=_cancel_subscription,
        risk=RiskLevel.HIGH,
        irreversible=True,
        effect="updates subscriptions.status and writes a cancellation record",
    )
)

register(
    ActionSpec(
        name="issue_partial_refund",
        description=(
            "Refund part of what a customer has already paid. This moves money, so policy "
            "always requires a human decision. Refunds above the policy ceiling additionally "
            "need an approver holding a supervisor or account-manager role."
        ),
        args_model=IssuePartialRefundArgs,
        fn=_issue_partial_refund,
        risk=RiskLevel.HIGH,
        irreversible=True,
        effect="inserts a row in refunds and logs a refund_issued ticket event",
    )
)

ALL_ACTIONS = ACTIONS
