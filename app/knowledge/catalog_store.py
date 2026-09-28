"""Product catalog source (``data/structured/catalog.csv``)."""

from __future__ import annotations

import csv
import logging
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Evidence, SourceType

logger = logging.getLogger(__name__)

_rows: list[dict[str, str]] | None = None


def load_catalog(force: bool = False, settings: Settings | None = None) -> list[dict[str, str]]:
    global _rows
    if _rows is not None and not force:
        return _rows
    settings = settings or get_settings()
    path = settings.catalog_path
    if not path.exists():
        raise FileNotFoundError(f"catalog file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        _rows = [dict(row) for row in csv.DictReader(handle)]
    return _rows


def _to_evidence(
    row: dict[str, str], query: str, label: str, score: float
) -> Evidence:
    rendered = ", ".join(f"{k}={v}" for k, v in row.items() if v not in ("", None))
    return Evidence(
        source_type=SourceType.CATALOG,
        source_name="catalog.csv",
        citation=f"catalog.csv :: {label}",
        snippet=rendered,
        score=score,
        query=query,
        metadata={k: v for k, v in row.items()},
    )


def catalog_evidence(query: str, top_k: int = 4) -> list[Evidence]:
    rows = load_catalog()
    terms = {t for t in query.lower().replace("?", " ").split() if len(t) > 2}

    scored: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        blob = " ".join(row.values()).lower()
        score = sum(1 for term in terms if term in blob)
        if score:
            scored.append((score, row))

    if not scored:
        scored = [(0, row) for row in rows[:top_k]]

    scored.sort(key=lambda pair: -pair[0])
    out: list[Evidence] = []
    for score, row in scored[:top_k]:
        label = f"{row.get('plan_name')} / {row.get('billing_cycle')}"
        out.append(_to_evidence(row, query, label, min(1.0, 0.3 + 0.2 * score)))
    return out


def plan_price(plan_name: str, billing_cycle: str = "monthly") -> dict[str, Any] | None:
    for row in load_catalog():
        if (
            row.get("plan_name", "").lower() == plan_name.lower()
            and row.get("billing_cycle", "").lower() == billing_cycle.lower()
        ):
            return row
    return None


def catalog_summary() -> dict[str, Any]:
    rows = load_catalog()
    plans = sorted({r.get("plan_name", "") for r in rows if r.get("plan_name")})
    return {"rows": len(rows), "plans": plans}
