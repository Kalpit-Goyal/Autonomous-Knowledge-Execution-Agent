"""Business policy source.

``data/structured/policies.json`` is the authority for SLA targets, refund
windows, escalation routing and approval requirements. The agent reads it at
runtime; no threshold in this module is a hardcoded business rule.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings, get_settings
from app.schemas import Evidence, SourceType

logger = logging.getLogger(__name__)

_cache: dict[str, Any] | None = None


def load_policies(force: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    global _cache
    if _cache is not None and not force:
        return _cache
    settings = settings or get_settings()
    path = settings.policies_path
    if not path.exists():
        raise FileNotFoundError(f"policies file not found: {path}")
    _cache = json.loads(path.read_text(encoding="utf-8"))
    return _cache


def _walk(node: Any, trail: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten the policy tree into (path, value) leaves."""
    out: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(_walk(value, trail + (str(key),)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            out.extend(_walk(value, trail + (str(i),)))
    else:
        out.append((trail, node))
    return out


def get_policy(section: str) -> dict[str, Any]:
    policies = load_policies()
    return policies.get(section, {})


def refund_window_days(plan: str, billing_cycle: str | None) -> int:
    """Refund window for a plan, honouring the annual-plan override.

    The KB article and this file disagree about the annual window on purpose.
    The caller is expected to surface both values rather than resolve the
    conflict silently, so this helper returns the policy value and the retrieval
    layer reports the disagreement.
    """
    refunds = get_policy("refunds")
    if (billing_cycle or "").lower() == "annual":
        return int(refunds.get("annual_plan_window_days", 0))
    return int(refunds.get("standard_window_days", 0))


def sla_for(priority: str) -> dict[str, Any]:
    tiers = get_policy("sla").get("tiers", {})
    return tiers.get((priority or "").upper(), {})


def requires_approval(action: str) -> bool:
    listed = get_policy("approvals").get("requires_human_approval_actions", [])
    return action in listed


def approval_rule() -> dict[str, Any]:
    return get_policy("approvals")


def escalation_targets(priority: str) -> dict[str, Any]:
    matrix = get_policy("escalation").get("matrix", {})
    return matrix.get((priority or "").upper(), {})


def policy_evidence(query: str, top_k: int = 4) -> list[Evidence]:
    """Rank policy leaves by lexical overlap with the query.

    Deliberately simple and deterministic: the policy file is small and flat
    enough that an exact term match is more trustworthy than an embedding, and
    it guarantees the numbers the LLM quotes are the ones in the file.
    """
    policies = load_policies()
    terms = {t for t in query.lower().replace("?", " ").split() if len(t) > 2}
    sections = {
        k: v
        for k, v in policies.items()
        if k not in {"conflicts_on_purpose", "version", "owner", "last_reviewed"}
    }

    scored: list[tuple[int, str, str, Any]] = []
    for section, value in sections.items():
        for trail, leaf in _walk(value):
            leaf_blob = json.dumps(leaf).lower()
            score = sum(1 for term in terms if term in leaf_blob)
            if score:
                dotted = ".".join((section, *trail))
                scored.append((score, section, dotted, leaf))

    scored.sort(key=lambda row: (-row[0], row[2]))
    grouped: list[Evidence] = []
    for score, section, dotted, leaf in scored[: top_k * 4]:
        citation = f"policies.json :: {dotted}"
        if any(e.citation == citation for e in grouped):
            continue
        grouped.append(
            Evidence(
                source_type=SourceType.POLICY,
                source_name="policies.json",
                citation=citation,
                snippet=json.dumps(leaf, indent=2) if not isinstance(leaf, str) else leaf,
                score=min(1.0, score / max(1, len(terms))),
                query=query,
                metadata={"section": section, "path": dotted, "value": leaf},
            )
        )
        if len(grouped) >= top_k:
            break

    if grouped:
        return grouped

    return [
        Evidence(
            source_type=SourceType.POLICY,
            source_name="policies.json",
            citation="policies.json :: (top level)",
            snippet=json.dumps(
                {k: (list(v.keys()) if isinstance(v, dict) else v) for k, v in sections.items()},
                indent=2,
            ),
            score=0.1,
            query=query,
            metadata={"fallback": True},
        )
    ]


def policy_summary() -> dict[str, Any]:
    policies = load_policies()
    return {
        "version": policies.get("version"),
        "last_reviewed": policies.get("last_reviewed"),
        "owner": policies.get("owner"),
        "sections": sorted(
            k
            for k in policies
            if k not in {"conflicts_on_purpose", "version", "owner", "last_reviewed"}
        ),
        "approver_roles": list(approval_rule().get("approver_roles", [])),
    }
