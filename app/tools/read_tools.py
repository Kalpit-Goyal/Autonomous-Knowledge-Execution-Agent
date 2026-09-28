"""Read-only retrieval tools.

Each tool takes a natural-language question and returns evidence with a
citation the LLM can quote. The tool bodies are thin: the work happens in the
knowledge layer, so the same access path serves the graph, the API and tests.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from app.knowledge import catalog_store, policy_store, sql_store
from app.knowledge.vector_store import ensure_indexed
from app.knowledge.vector_store import search as kb_search
from app.schemas import Evidence, SourceType

logger = logging.getLogger(__name__)

ToolFn = Callable[..., list[Evidence]]


def search_knowledge_base(query: str, top_k: int = 5, source_filter: str = "") -> list[Evidence]:
    ensure_indexed()
    hits = kb_search(query, top_k=top_k)
    if source_filter:
        needle = source_filter.strip().lower()
        hits = [h for h in hits if needle in h.source_name.lower() or needle in h.citation.lower()]
    return hits


def query_operations_data(question: str, sql: str = "") -> list[Evidence]:
    """Answer a factual question about customers, subscriptions or tickets.

    When the caller supplies SQL it is validated by
    :func:`app.knowledge.sql_store.validate_sql` before it runs. When it does
    not, the question is turned into a query by the LLM in the retrieval node
    and passed here - this tool never invents SQL on its own.
    """
    if not sql:
        return []
    rows, statement = sql_store.execute_readonly(sql)
    if not rows:
        return []
    columns = list(rows[0].keys())
    body = "\n".join(" | ".join(f"{k}={r.get(k)}" for k in columns) for r in rows[:20])
    if len(rows) > 20:
        body += "\n... (truncated)"
    return [
        Evidence(
            source_type=SourceType.OPS_DB,
            source_name="support.db",
            citation=f"support.db ({len(rows)} rows: {', '.join(columns)})",
            snippet=body,
            score=1.0,
            query=question,
            metadata={"row_count": len(rows), "sql": statement, "columns": columns},
        )
    ]


def lookup_business_policy(question: str) -> list[Evidence]:
    return policy_store.policy_evidence(question, top_k=5)


def lookup_product_catalog(question: str) -> list[Evidence]:
    return catalog_store.catalog_evidence(question, top_k=4)


def describe_schema() -> list[Evidence]:
    schema = sql_store.schema_description()
    return [
        Evidence(
            source_type=SourceType.OPS_DB,
            source_name="support.db",
            citation="support.db :: schema",
            snippet=schema,
            score=1.0,
            query="schema",
            metadata={"schema": schema},
        )
    ]


TOOL_FOR_SOURCE: dict[SourceType, str] = {
    SourceType.KB: "search_knowledge_base",
    SourceType.POLICY: "lookup_business_policy",
    SourceType.CATALOG: "lookup_product_catalog",
    SourceType.OPS_DB: "query_operations_data",
}


def tool_for_source(source_type: SourceType) -> str:
    return TOOL_FOR_SOURCE.get(source_type, "search_knowledge_base")


READ_TOOLS: dict[str, dict[str, Any]] = {
    "search_knowledge_base": {
        "fn": search_knowledge_base,
        "description": (
            "Semantic search over the product and support knowledge base (markdown articles). "
            "Use for policy explanations, procedures, playbooks, and documented behaviour."
        ),
        "parameters": {
            "query": {"type": "string", "description": "Natural language question or topic."},
            "top_k": {"type": "integer", "description": "How many passages to return (default 5)."},
        },
        "required": ["query"],
    },
    "query_operations_data": {
        "fn": query_operations_data,
        "description": (
            "Run one read-only SELECT against the operations database (customers, subscriptions, "
            "tickets) and return the rows. Use for account state, ARR, renewal dates, ticket "
            "history, open counts. Never use this to modify anything."
        ),
        "parameters": {
            "question": {"type": "string", "description": "What the query is trying to find out."},
            "sql": {
                "type": "string",
                "description": "A single SELECT statement against the operations schema.",
            },
        },
        "required": ["question", "sql"],
    },
    "lookup_business_policy": {
        "fn": lookup_business_policy,
        "description": (
            "Read the authoritative business policy values: SLA targets, refund windows, "
            "escalation routing, approval thresholds, entitlement limits. When policy and the "
            "knowledge base disagree, this source wins and both values should be reported."
        ),
        "parameters": {
            "question": {
                "type": "string",
                "description": "The policy question, e.g. 'refund window'.",
            }
        },
        "required": ["question"],
    },
    "lookup_product_catalog": {
        "fn": lookup_product_catalog,
        "description": (
            "Read the product catalog: plan names, billing cycles, prices, included seats, "
            "storage, API call allowances and support tiers. Use for any pricing or entitlement "
            "question."
        ),
        "parameters": {
            "question": {"type": "string", "description": "Plan or pricing question."}
        },
        "required": ["question"],
    },
    "describe_operations_schema": {
        "fn": describe_schema,
        "description": (
            "List the tables and columns available in the operations database. Call this before "
            "writing SQL for query_operations_data if you are unsure of the column names."
        ),
        "parameters": {},
        "required": [],
    },
}


def read_tool_names() -> list[str]:
    return list(READ_TOOLS)


def run_read_tool(name: str, **kwargs: Any) -> list[Evidence]:
    spec = READ_TOOLS.get(name)
    if spec is None:
        return [
            Evidence(
                source_type=SourceType.KB,
                source_name="unknown",
                citation=f"tool:{name}",
                snippet=f"unknown read tool: {name}",
                score=0.0,
            )
        ]
    fn: ToolFn = spec["fn"]
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("read tool %s failed: %s", name, exc)
        return [
            Evidence(
                source_type=SourceType.KB,
                source_name=name,
                citation=f"tool:{name} (failed)",
                snippet=f"{type(exc).__name__}: {exc}",
                score=0.0,
            )
        ]


def describe_read_tools() -> str:
    lines: list[str] = []
    for name, spec in READ_TOOLS.items():
        lines.append(f"{name}: {spec['description']}")
        if spec["parameters"]:
            lines.append("  parameters: " + ", ".join(spec["parameters"]))
    return "\n".join(lines)
