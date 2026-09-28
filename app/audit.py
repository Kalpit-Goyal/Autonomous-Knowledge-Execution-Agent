"""Audit trail.

Every retrieval, decision, approval and action writes one row here. The table is
the evidence the UI reads, and it is the same table the tests assert against.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.knowledge.sql_store import get_connection

logger = logging.getLogger(__name__)


def _merge(payload: Any, data: Any) -> Any:
    """``data`` is a friendlier alias for ``payload``; combine rather than drop."""
    if payload is None:
        return data
    if data is None:
        return payload
    if isinstance(payload, dict) and isinstance(data, dict):
        return {**payload, **data}
    return {"payload": payload, "data": data}


def _dumps(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def record(
    event_type: str,
    reasoning: str = "",
    *,
    node: str = "",
    session_id: str | None = None,
    action: str | None = None,
    actor: str = "agent",
    payload: Any = None,
    data: Any = None,
    result: Any = None,
    evidence: Any = None,
    duration_ms: int = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
    ok: bool = True,
    error: str | None = None,
    settings: Settings | None = None,
) -> int:
    conn = get_connection(settings)
    try:
        cursor = conn.execute(
            "INSERT INTO audit_log (session_id, node, event_type, action, actor, payload, "
            "result, evidence, reasoning, duration_ms, tokens_in, tokens_out, ok, error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                node,
                event_type,
                action,
                actor,
                _dumps(_merge(payload, data)),
                _dumps(result),
                _dumps(evidence),
                reasoning,
                int(duration_ms),
                int(tokens_in),
                int(tokens_out),
                1 if ok else 0,
                error,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)
    except Exception as exc:  # noqa: BLE001
        logger.error("audit write failed: %s", exc)
        return 0
    finally:
        conn.close()


def query(
    *,
    session_id: str | None = None,
    event_type: str | None = None,
    action: str | None = None,
    limit: int = 100,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    if action:
        clauses.append("action = ?")
        params.append(action)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))

    conn = get_connection(settings)
    try:
        rows = conn.execute(
            f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def stats(settings: Settings | None = None) -> dict[str, Any]:
    conn = get_connection(settings)
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM audit_log").fetchone()["n"]
        by_type = conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM audit_log GROUP BY event_type ORDER BY n DESC"
        ).fetchall()
        failures = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE ok = 0"
        ).fetchone()["n"]
    finally:
        conn.close()
    return {
        "total": int(total),
        "failures": int(failures),
        "by_event_type": {r["event_type"]: int(r["n"]) for r in by_type},
    }
