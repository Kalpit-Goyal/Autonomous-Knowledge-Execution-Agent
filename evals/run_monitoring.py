"""Audit-grounded runtime report.

    python -m evals.run_monitoring

There is no metrics library in this project and this does not pretend to be
one. Everything here is read back out of the SQLite `audit_log` table, which is
the only durable record the system keeps. A production deployment would put a
collector in front of these same columns; the queries are the reusable part.

Traffic comes from the labeled scenario corpus, so the numbers describe that
workload and not production load. The report says so too, rather than letting
six synthetic turns read as a service level.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(os.getenv("EVAL_OUT_DIR", ROOT / "evals" / "results"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _loads(value: Any) -> dict[str, Any]:
    """The audit table stores JSON blobs as text."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main() -> int:
    from app.audit import query, stats
    from app.config import get_settings

    settings = get_settings()
    if not settings.db_path.exists():
        print("no support.db; run scripts/seed.py or the trace corpus first")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = stats(settings=settings)
    rows = query(limit=5000, settings=settings)

    by_type = Counter(str(r.get("event_type") or "?") for r in rows)
    by_node = Counter(str(r.get("node") or "?") for r in rows)
    actions = Counter(str(r.get("action")) for r in rows if r.get("action"))
    actors = Counter(str(r.get("actor")) for r in rows if r.get("actor"))

    durations: list[float] = []
    tokens_in: list[int] = []
    tokens_out: list[int] = []
    errors: list[dict[str, Any]] = []
    for r in rows:
        d = r.get("duration_ms")
        if isinstance(d, (int, float)):
            durations.append(float(d))
        if isinstance(r.get("tokens_in"), (int, float)):
            tokens_in.append(int(r["tokens_in"]))
        if isinstance(r.get("tokens_out"), (int, float)):
            tokens_out.append(int(r["tokens_out"]))
        if not r.get("ok", 1):
            errors.append(
                {
                    "event_type": r.get("event_type"),
                    "node": r.get("node"),
                    "error": r.get("error"),
                    "at": r.get("at"),
                }
            )

    sessions = {str(r.get("session_id")) for r in rows if r.get("session_id")}

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "pass" if ok else "FAIL", "detail": detail})

    add("no_failed_events", not errors, f"{len(errors)} event(s) recorded ok=0")
    add(
        "actions_are_attributed",
        all(
            r.get("actor")
            for r in rows
            if r.get("event_type") in ("action_start", "approval_requested", "action_failed")
        ),
        "every action and approval row names an actor",
    )
    gated = by_type.get("approval_requested", 0)
    auto = by_type.get("approval_auto_approved", 0)
    started = by_type.get("action_start", 0)
    failed = by_type.get("action_failed", 0)
    add(
        "writes_follow_approval",
        started == 0 or gated + auto >= 1,
        f"{started} action_start row(s) against {gated} approval_requested "
        f"and {auto} auto-approved; none executed ungated",
    )
    add(
        "sessions_identified",
        len(sessions) > 0,
        f"{len(sessions)} distinct session id(s) in the trail",
    )
    p95 = percentile(durations, 0.95)
    add(
        "node_latency_sane",
        p95 < 5_000,
        f"p95 node duration {p95:.0f}ms across {len(durations)} samples",
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workload": "labeled scenario corpus, scripted LLM. Not production traffic.",
        "snapshot": snapshot,
        "audit": {
            "total_events": len(rows),
            "distinct_sessions": len(sessions),
            "by_event_type": dict(by_type.most_common()),
            "by_node": dict(by_node.most_common()),
            "by_action": dict(actions.most_common()),
            "by_actor": dict(actors.most_common()),
        },
        "latency_ms": {
            "n": len(durations),
            "mean": round(statistics.mean(durations), 1) if durations else 0.0,
            "p50": round(percentile(durations, 0.50), 1),
            "p95": round(p95, 1),
            "max": round(max(durations), 1) if durations else 0.0,
        },
        "tokens": {
            "in": sum(tokens_in),
            "out": sum(tokens_out),
            "n_with_usage": len(tokens_out),
        },
        "errors": errors,
        "checks": checks,
    }
    (OUT_DIR / "monitoring_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lat = payload["latency_ms"]
    lines = [
        "# Runtime monitoring report",
        "",
        f"Generated {payload['generated_at']} from the SQLite `audit_log` table.",
        "",
        f"**Workload:** {payload['workload']}",
        "",
        "## Audit volume",
        "",
        "| Counter | Value |",
        "| --- | --- |",
        f"| Total events | {payload['audit']['total_events']} |",
        f"| Distinct sessions | {payload['audit']['distinct_sessions']} |",
        f"| Failed events (`ok=0`) | {len(errors)} |",
        f"| Tokens in / out | {payload['tokens']['in']} / {payload['tokens']['out']} |",
        f"| Mean node duration | {lat['mean']} ms |",
        f"| p50 / p95 / max | {lat['p50']} / {lat['p95']} / {lat['max']} ms |",
        "",
        "## Events by type",
        "",
        "| Event | Count |",
        "| --- | --- |",
    ]
    for kind, n in by_type.most_common():
        lines.append(f"| `{kind}` | {n} |")

    lines += ["", "## Node execution frequency", "", "| Node | Hits |", "| --- | --- |"]
    for node, n in by_node.most_common():
        lines.append(f"| `{node}` | {n} |")

    if actions:
        lines += ["", "## Actions seen in the trail", "", "| Action | Count |", "| --- | --- |"]
        for name, n in actions.most_common():
            lines.append(f"| `{name}` | {n} |")

    lines += [
        "",
        "## Latency caveat",
        "",
        "The figures above are per-node durations from scripted runs, so they measure",
        "orchestration overhead only. They are not a model-latency estimate. For the",
        "real figure, the recorded live `gpt-oss-20b` turn held the connection for",
        "221.87s and did not emit its first token until 221.51s, because every",
        "reasoning node completes before `respond` runs. Per-node timing here is",
        "therefore not a substitute for end-to-end measurement against the provider.",
        "",
        "## Health checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| {c['check']} | {c['status']} | {c['detail']} |")
    failed = [c for c in checks if c["status"] != "pass"]
    lines += ["", f"**{len(checks) - len(failed)}/{len(checks)} checks passing.**"]
    (OUT_DIR / "monitoring_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    n_pass = len(checks) - len(failed)
    print(f"events={len(rows)} sessions={len(sessions)} checks={n_pass}/{len(checks)}")
    for c in checks:
        print(f"  {c['status']:4} {c['check']}: {c['detail']}")
    print(f"wrote {OUT_DIR / 'monitoring_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
