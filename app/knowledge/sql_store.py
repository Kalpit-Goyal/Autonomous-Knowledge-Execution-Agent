"""Operational store: SQLite over customers, subscriptions and tickets.

The database is seeded from the CSV files in ``data/structured`` by
:mod:`scripts.seed`. The agent reads it through a validated read-only SQL path -
the model proposes a query, :func:`execute_readonly` decides whether it is
actually safe to run.
"""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Evidence, SourceType

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id   TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL,
    domain        TEXT,
    plan          TEXT,
    seats_used    INTEGER DEFAULT 0,
    region        TEXT,
    health        TEXT,
    arr_usd       REAL DEFAULT 0,
    renewal_date  TEXT,
    crm_owner     TEXT
);

CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    plan            TEXT,
    billing_cycle   TEXT,
    status          TEXT,
    mrr_usd         REAL DEFAULT 0,
    renewal_date    TEXT,
    started_at      TEXT,
    auto_renew      INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id    TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL,
    subject      TEXT,
    priority     TEXT,
    status       TEXT,
    created_at   TEXT,
    updated_at   TEXT,
    open_minutes INTEGER DEFAULT 0,
    category     TEXT,
    sentiment    TEXT
);

CREATE TABLE IF NOT EXISTS ticket_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    body       TEXT,
    actor      TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS account_credits (
    credit_id   TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    amount_usd  REAL NOT NULL,
    reason      TEXT,
    approved_by TEXT,
    expires_at  TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id     TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    subscription_id TEXT,
    amount_usd    REAL NOT NULL,
    reason        TEXT,
    approved_by   TEXT,
    approval_role TEXT,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT,
    status     TEXT,
    summary    TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS approvals (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    action        TEXT NOT NULL,
    args_json     TEXT,
    why           TEXT,
    risk          TEXT,
    reason        TEXT,
    evidence_json TEXT,
    status        TEXT DEFAULT 'pending',
    decided_by    TEXT,
    approver_role TEXT,
    decision_note TEXT,
    created_at    TEXT DEFAULT (datetime('now')),
    decided_at    TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    node        TEXT,
    event_type  TEXT,
    action      TEXT,
    actor       TEXT,
    payload     TEXT,
    result      TEXT,
    evidence    TEXT,
    reasoning   TEXT,
    duration_ms INTEGER DEFAULT 0,
    tokens_in   INTEGER DEFAULT 0,
    tokens_out  INTEGER DEFAULT 0,
    ok          INTEGER DEFAULT 1,
    error       TEXT,
    at          TEXT DEFAULT (datetime('now'))
);
"""

SEED_FILES = {
    "customers": "customers.csv",
    "subscriptions": "subscriptions.csv",
    "tickets": "tickets.csv",
}

NUMERIC_COLUMNS = {
    "customers": {"seats_used", "arr_usd"},
    "subscriptions": {"mrr_usd", "auto_renew"},
    "tickets": {"open_minutes"},
}

# Tables the agent itself writes to. They have no CSV seed, so `seed --reset`
# truncates them separately.
RUNTIME_TABLES = (
    "ticket_events",
    "account_credits",
    "refunds",
    "sessions",
    "messages",
    "approvals",
    "audit_log",
)

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|"
    r"reindex|begin|commit|rollback|grant|revoke|truncate|upsert)\b",
    re.IGNORECASE,
)
_SELECT_START = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_COMMENT_START = re.compile(r"^\s*(--|/\*|\*)")

_lock = threading.RLock()


def _connect(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(settings: Settings | None = None) -> None:
    with _lock, _connect(settings) as conn:
        conn.executescript(DDL)
        conn.commit()


def is_seeded(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if not settings.db_path.exists():
        return False
    with _lock, _connect(settings) as conn:
        try:
            row = conn.execute("SELECT COUNT(*) AS n FROM tickets").fetchone()
        except sqlite3.Error:
            return False
        return int(row["n"]) > 0


def seed_from_csv(
    *, reset: bool = False, settings: Settings | None = None
) -> dict[str, int]:
    settings = settings or get_settings()
    init_db(settings)
    counts: dict[str, int] = {}

    with _lock, _connect(settings) as conn:
        if reset:
            # Clear agent-written history too, so `--reset` is a clean slate
            # rather than only restoring the reference rows.
            for table in RUNTIME_TABLES:
                conn.execute(f"DELETE FROM {table}")
        for table, filename in SEED_FILES.items():
            path: Path = settings.structured_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"missing seed file: {path}")
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            if not rows:
                counts[table] = 0
                continue
            columns = list(rows[0].keys())
            numeric = NUMERIC_COLUMNS.get(table, set())
            placeholders = ", ".join("?" for _ in columns)
            collist = ", ".join(columns)
            payload = []
            for row in rows:
                payload.append(
                    tuple(
                        int(row[c]) if c in numeric and row.get(c) not in (None, "") else row[c]
                        for c in columns
                    )
                )
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})", payload
            )
            counts[table] = len(payload)
        conn.commit()
    return counts


def table_names(settings: Settings | None = None) -> list[str]:
    with _lock, _connect(settings) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


MODEL_VISIBLE_TABLES = (
    "customers",
    "subscriptions",
    "tickets",
    "ticket_events",
    "account_credits",
    "refunds",
)


def schema_description(settings: Settings | None = None) -> str:
    """Compact schema summary handed to the LLM when it writes SQL.

    Deliberately limited to the operational tables. ``audit_log``,
    ``approvals`` and ``messages`` hold agent-internal bookkeeping and are
    exposed to the model only through dedicated tools, not free-form SQL.
    """
    lines: list[str] = []
    for table in MODEL_VISIBLE_TABLES:
        columns = _columns_for(table, settings)
        if not columns:
            continue
        rendered = ", ".join(
            f"{c['name']} {c['type'] or 'TEXT'}" for c in columns if c["name"]
        )
        lines.append(f"{table}({rendered})")
    return "\n".join(lines)


def row_counts(settings: Settings | None = None) -> dict[str, int]:
    """Row count per table, for the diagnostics endpoint."""
    counts: dict[str, int] = {}
    with _lock, _connect(settings) as conn:
        for name in table_names(settings):
            try:
                counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            except sqlite3.Error:
                counts[name] = -1
    return counts


def _columns_for(table: str, settings: Settings | None = None) -> list[dict[str, Any]]:
    with _lock, _connect(settings) as conn:
        return [dict(r) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def validate_sql(sql: str, *, max_rows: int = 50) -> tuple[bool, str]:
    """Accept only a single read-only SELECT. Returns ``(ok, reason)``."""
    statement = (sql or "").strip().rstrip(";").strip()
    if not statement:
        return False, "empty statement"
    if _COMMENT_START.match(statement):
        return False, "statement must not start with a comment"
    if not _SELECT_START.match(statement):
        return False, "only SELECT (or WITH ... SELECT) statements are permitted"
    if ";" in statement:
        return False, "only a single statement is permitted"
    forbidden = FORBIDDEN.search(statement)
    if forbidden:
        return False, f"keyword '{forbidden.group(0).upper()}' is not allowed in a read-only query"
    lowered = statement.lower()
    if not re.search(r"\blimit\b", lowered):
        statement = f"{statement} LIMIT {int(max_rows)}"
    return True, statement


def execute_readonly(
    sql: str, *, settings: Settings | None = None, max_rows: int | None = None
) -> tuple[list[dict[str, Any]], str]:
    settings = settings or get_settings()
    limit = max_rows or settings.max_sql_rows
    ok, statement = validate_sql(sql, max_rows=limit)
    if not ok:
        return [], f"rejected: {statement}"

    with _lock, _connect(settings) as conn:
        conn.execute("PRAGMA query_only=ON")
        try:
            cursor = conn.execute(statement)
            rows = [dict(r) for r in cursor.fetchall()]
        except sqlite3.Error as exc:
            return [], f"sql error: {exc}"
    return rows[:limit], statement


def query_evidence(
    sql: str, purpose: str = "", *, settings: Settings | None = None, max_rows: int | None = None
) -> list[Evidence]:
    rows, statement = execute_readonly(sql, settings=settings, max_rows=max_rows)
    if not rows:
        return []
    columns = list(rows[0].keys())
    rendered = "\n".join(
        " | ".join(f"{k}={row.get(k)}" for k in columns) for row in rows[:20]
    )
    truncated = len(rows) > 20
    body = rendered + ("\n... (truncated)" if truncated else "")
    return [
        Evidence(
            source_type=SourceType.OPS_DB,
            source_name="support.db",
            citation=f"ops_db://{', '.join(columns)} ({len(rows)} rows)",
            snippet=body,
            score=1.0,
            query=purpose or statement,
            metadata={"row_count": len(rows), "sql": statement, "columns": columns},
        )
    ]


def write(query: str, params: tuple | dict | None = None, settings: Settings | None = None):
    """Internal write helper used by the action layer (never model-generated SQL)."""
    with _lock, _connect(settings) as conn:
        cursor = conn.execute(query, params or ())
        conn.commit()
        return cursor


def get_connection(settings: Settings | None = None) -> sqlite3.Connection:
    return _connect(settings)
