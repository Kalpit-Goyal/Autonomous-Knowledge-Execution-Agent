"""Durable index of approval requests.

The graph checkpoint holds the authoritative state of a paused run, but it is
awkward to query: a checkpoint is addressed by thread id and holds a serialised
blob. This table is a flat, queryable mirror of the same facts, so the API can
answer "what is waiting for a human" without walking every checkpoint, and so a
pending approval survives a process restart.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.knowledge.sql_store import get_connection, write
from app.schemas import ApprovalRequest

logger = logging.getLogger(__name__)


def record_request(request: ApprovalRequest, settings: Settings | None = None) -> None:
    write(
        "INSERT OR REPLACE INTO approvals (id, session_id, action, args_json, why, risk, reason, "
        "evidence_json, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            request.id,
            request.session_id,
            request.action,
            json.dumps(request.args, default=str),
            request.why,
            request.risk.value,
            request.reason_for_approval,
            json.dumps(request.evidence_refs),
            request.status,
            request.created_at,
        ),
        settings,
    )


def record_decision(
    approval_id: str,
    *,
    approved: bool,
    approver: str,
    reason: str,
    decided_at: str,
    approver_role: str = "",
    settings: Settings | None = None,
) -> None:
    write(
        "UPDATE approvals SET status = ?, decided_by = ?, approver_role = ?, "
        "decision_note = ?, decided_at = ? WHERE id = ?",
        (
            "approved" if approved else "rejected",
            approver,
            approver_role,
            reason,
            decided_at,
            approval_id,
        ),
        settings,
    )


def record_auto_approval(request: ApprovalRequest, settings: Settings | None = None) -> None:
    record_request(request, settings)
    record_decision(
        request.id,
        approved=True,
        approver=request.decided_by or "auto_approve_policy",
        reason=request.decision_reason or "",
        decided_at=request.decided_at or "",
        settings=settings,
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("args_json", "evidence_json"):
        raw = data.pop(key, None)
        if raw:
            try:
                data[key.replace("_json", "")] = json.loads(raw)
            except (TypeError, ValueError):
                data[key.replace("_json", "")] = raw
    return data


def pending(settings: Settings | None = None, limit: int = 50) -> list[dict[str, Any]]:
    conn = get_connection(settings)
    try:
        rows = conn.execute(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def for_session(session_id: str, settings: Settings | None = None) -> list[dict[str, Any]]:
    conn = get_connection(settings)
    try:
        rows = conn.execute(
            "SELECT * FROM approvals WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def recent(limit: int = 50, settings: Settings | None = None) -> list[dict[str, Any]]:
    conn = get_connection(settings)
    try:
        rows = conn.execute(
            "SELECT * FROM approvals ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()
