"""Graph nodes.

Each function here is one reasoning step. They share three conventions:

* They return plain dicts, never mutating ``state`` in place, because the state
  is checkpointed and replayed.
* Read tools and actions run through a thread pool, so independent lookups and
  independent writes genuinely overlap rather than being awaited one by one.
* Every node appends a trace step, and side effects go through the audit log, so
  the UI can show what happened without re-deriving it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app import approval_store
from app.audit import record
from app.config import get_settings
from app.graph import prompts
from app.graph.state import AgentState
from app.knowledge import sql_store
from app.knowledge.policy_store import approval_rule
from app.schemas import (
    ActionOutcome,
    ApprovalRequest,
    Decision,
    DecisionSet,
    Evidence,
    Intake,
    Plan,
    PlanStep,
    ReconcileStatus,
    Reconciliation,
    RetrievalTask,
    SourceType,
    TraceStep,
    Verification,
    utcnow_iso,
)
from app.tools import (
    ActionContext,
    get_action,
    needs_approval,
    run_read_tool,
    validate_args,
)
from app.tools.read_tools import tool_for_source

logger = logging.getLogger(__name__)

MAX_PARALLEL = 6


def _trace(
    node: str, label: str, detail: str = "", data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Trace steps are stored as plain dicts so the checkpoint stays serialisable."""
    return TraceStep(node=node, label=label, detail=detail, data=data or {}).model_dump(mode="json")


def _dump_evidence(items: list[Evidence]) -> list[dict[str, Any]]:
    return [e.model_dump(mode="json") for e in items]


def _load_evidence(items: list[dict[str, Any]]) -> list[Evidence]:
    return [Evidence.model_validate(i) for i in items]


def _dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Evidence] = []
    for ev in items:
        key = (ev.source_type.value, ev.citation, ev.snippet[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _render_evidence(evidence: list[Evidence], limit: int = 24) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    blocks: list[str] = []
    for ev in evidence[:limit]:
        snippet = ev.snippet if len(ev.snippet) <= 1200 else ev.snippet[:1200] + " ...[cut]"
        blocks.append(
            f"id={ev.id} source={ev.source_type.value} score={ev.score:.3f}\n"
            f"  citation: {ev.citation}\n  content: {snippet}"
        )
    if len(evidence) > limit:
        blocks.append(f"... and {len(evidence) - limit} more passages")
    return "\n".join(blocks)


def _run_tasks_in_parallel(
    tasks: list[Callable[[], Any]], max_workers: int = MAX_PARALLEL
) -> list[Any]:
    if not tasks:
        return []
    if len(tasks) == 1:
        return [tasks[0]()]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        futures = [pool.submit(fn) for fn in tasks]
        results: list[Any] = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("parallel task failed: %s", exc)
                results.append(exc)
        return results


# --------------------------------------------------------------------------- #
# intake
# --------------------------------------------------------------------------- #


def intake(state: AgentState, *, llm, memory) -> dict[str, Any]:
    recalled: list[dict[str, Any]] = []
    try:
        for item in memory.recall(
            "turn", session_id=state["session_id"], query=state["message"], limit=3
        ):
            recalled.append(
                {"key": item.key, "value": item.value, "score": item.score, "at": item.updated_at}
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory recall failed: %s", exc)

    schema_block = prompts.ops_schema_block()
    human = (
        f"<request>\n{state['message']}\n</request>\n\n"
        f"{prompts.knowledge_base_prompt()}\n{schema_block}\n"
        f"recalled context from earlier in this session:\n"
        f"{json.dumps(recalled, default=str) if recalled else '(none)'}\n"
    )
    parsed, tin, tout = llm.structured(
        Intake, system=prompts.INTAKE_PROMPT, human=human, node="intake"
    )

    record(
        "intake",
        f"intent={parsed.intent.value} customer={parsed.customer_id or '-'} "
        f"ticket={parsed.ticket_id or '-'} state_change={parsed.wants_state_change}",
        session_id=state["session_id"],
        node="intake",
        data={"intake": parsed.model_dump(mode="json"), "memory_hits": len(recalled)},
    )

    return {
        "intake": parsed.model_dump(mode="json"),
        "memory_hits": recalled,
        "tokens_in": state.get("tokens_in", 0) + tin,
        "tokens_out": state.get("tokens_out", 0) + tout,
        "trace": [
            _trace(
                "intake",
                "Understood the request",
                parsed.summary,
                {
                    "intent": parsed.intent.value,
                    "customer_id": parsed.customer_id,
                    "ticket_id": parsed.ticket_id,
                    "wants_state_change": parsed.wants_state_change,
                    "open_points": parsed.open_points,
                    "memory_recalled": len(recalled),
                },
            )
        ],
    }


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def plan(state: AgentState, *, llm) -> dict[str, Any]:
    prior = ""
    if state.get("verification"):
        prior = (
            "\nA previous attempt was not verified as complete. Here is the verification "
            f"report:\n{json.dumps(state['verification'], default=str)}\n\n"
            "Produce a revised plan that closes those gaps. Do not repeat work that already "
            "succeeded.\n"
        )
    elif state.get("iteration", 0) > 0:
        prior = "\nThis is a re-plan after new evidence. Focus on what is still unknown.\n"

    human = (
        f"<request>\n{state['message']}\n</request>\n\n"
        f"<intake>\n{json.dumps(state.get('intake') or {}, default=str)}\n</intake>\n\n"
        f"{prior}\n{schema_block_for_planner()}\n"
    )
    parsed, tin, tout = llm.structured(
        Plan, system=prompts.PLANNER_PROMPT, human=human, node="plan"
    )

    # The planner prompt shows the verification report, but a follow-up query
    # the LLM did not echo back must still be retrieved.
    plan_dict = parsed.model_dump(mode="json")
    if state.get("_follow_up_queries"):
        plan_dict = append_followup_tasks(
            plan_dict,
            list(state["_follow_up_queries"]),
            why="The verification step found the goal was not yet met.",
            step_id="verify_followup",
        )
    parsed = Plan.model_validate(plan_dict)

    tasks: list[RetrievalTask] = []
    for step in parsed.steps:
        step.tasks = [t.model_copy(update={"step_id": step.id}) for t in step.tasks]
        tasks.extend(step.tasks)

    record(
        "plan",
        f"goal={parsed.goal!r} steps={len(parsed.steps)} tasks={len(tasks)}",
        session_id=state["session_id"],
        node="plan",
        data={"plan": parsed.model_dump(mode="json")},
    )

    return {
        "plan": parsed.model_dump(mode="json"),
        "tokens_in": state.get("tokens_in", 0) + tin,
        "tokens_out": state.get("tokens_out", 0) + tout,
        "trace": [
            _trace(
                "plan",
                f"Planned {len(parsed.steps)} step(s)",
                parsed.goal,
                {
                    "goal": parsed.goal,
                    "steps": [
                        {
                            "id": s.id,
                            "goal": s.goal,
                            "why": s.why,
                            "sources": [x.value for x in s.sources],
                            "tasks": [
                                {"id": t.step_id, "source": t.source_type.value, "query": t.query}
                                for t in s.tasks
                            ],
                        }
                        for s in parsed.steps
                    ],
                    "open_questions": parsed.open_questions,
                },
            )
        ],
    }


def schema_block_for_planner() -> str:
    return prompts.ops_schema_block()


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #


def retrieve(state: AgentState, *, llm) -> dict[str, Any]:
    plan_obj = Plan.model_validate(state["plan"])
    tasks = [t for step in plan_obj.steps for t in step.tasks]
    if not tasks:
        return {
            "evidence": [],
            "trace": [_trace("retrieve", "Nothing to retrieve", "the plan produced no tasks")],
        }

    resolved = _ensure_sql(tasks, state, llm)
    calls: list[dict[str, Any]] = []

    def make_call(task: RetrievalTask) -> Callable[[], list[Evidence]]:
        def call() -> list[Evidence]:
            tool = task.tool or tool_for_source(task.source_type)
            kwargs: dict[str, Any] = {}
            if tool == "search_knowledge_base":
                kwargs = {"query": task.query, "top_k": 5}
            elif tool == "lookup_business_policy" or tool == "lookup_product_catalog":
                kwargs = {"question": task.query}
            elif tool == "query_operations_data":
                kwargs = {"question": task.query, "sql": task.sql}
            started = time.perf_counter()
            items = run_read_tool(tool, **kwargs)
            calls.append(
                {
                    "step_id": task.step_id,
                    "tool": tool,
                    "query": task.query,
                    "sql": task.sql,
                    "purpose": task.purpose,
                    "hits": len(items),
                    "ms": int((time.perf_counter() - started) * 1000),
                }
            )
            return items

        return call

    batches = _run_tasks_in_parallel([make_call(t) for t in resolved])
    evidence: list[Evidence] = []
    failures: list[str] = []
    for result in batches:
        if isinstance(result, Exception):
            failures.append(f"{type(result).__name__}: {result}")
            continue
        evidence.extend(result)

    for ev in evidence:
        ev.metadata.setdefault("step_id", "")
    # Union with what earlier iterations already retrieved: the verify loop
    # re-enters this node, and discarding the first pass would make the
    # reconciler and the verifier reason over less than we know.
    deduped = _dedupe_evidence(_load_evidence(state.get("evidence") or []) + evidence)
    deduped.sort(key=lambda e: e.score, reverse=True)

    record(
        "retrieve",
        f"tasks={len(resolved)} evidence={len(deduped)}",
        session_id=state["session_id"],
        node="retrieve",
        data={"calls": calls},
    )

    detail = f"{len(deduped)} passages from {len(resolved)} concurrent lookups"
    if failures:
        detail += f"; {len(failures)} failed"
    return {
        "evidence": _dump_evidence(deduped),
        "retrieved_tool_calls": calls,
        "trace": [
            _trace(
                "retrieve",
                f"Retrieved {len(deduped)} passage(s)",
                detail,
                {
                    "calls": [
                        f"{c['tool']} <- {c['query'][:70]} ({c['hits']} hits, {c['ms']}ms)"
                        for c in calls
                    ],
                    "failures": failures,
                },
            )
        ],
    }


def _ensure_sql(
    tasks: list[RetrievalTask], state: AgentState, llm
) -> list[RetrievalTask]:
    """Fill in missing ops_db SQL, since a query without SQL cannot run."""
    out: list[RetrievalTask] = []
    for task in tasks:
        if task.source_type is not SourceType.OPS_DB or task.sql.strip():
            out.append(task)
            continue
        parsed, _, _ = llm.structured(
            SqlQuery,
            system=SQL_WRITER_PROMPT,
            human=(
                f"schema:\n{prompts.ops_schema_block()}\n\n"
                f"what we need to find out: {task.query}\n"
                f"purpose: {task.purpose}\n"
            ),
            node="sql_writer",
        )
        ok, detail = sql_store.validate_sql(parsed.sql)
        if not ok:
            logger.warning("generated SQL rejected (%s): %s", detail, parsed.sql)
            out.append(task.model_copy(update={"sql": "SELECT 1 WHERE 0"}))
            continue
        out.append(task.model_copy(update={"sql": parsed.sql}))
    return out


# --------------------------------------------------------------------------- #
# reconciliation
# --------------------------------------------------------------------------- #


def reconcile(state: AgentState, *, llm) -> dict[str, Any]:
    evidence = _load_evidence(state.get("evidence") or [])
    human = (
        f"<request>\n{state['message']}\n</request>\n\n"
        f"<intake>\n{json.dumps(state.get('intake') or {}, default=str)}\n</intake>\n\n"
        f"<retrieved evidence>\n{_render_evidence(evidence)}\n</retrieved evidence>\n"
    )
    parsed, tin, tout = llm.structured(
        Reconciliation, system=prompts.RECONCILER_PROMPT, human=human, node="reconcile"
    )

    conflicts = [
        {"summary": text, "iteration": state.get("iteration", 0)} for text in parsed.conflicts
    ]
    # Accumulate: a conflict found on pass one is still worth reporting even if
    # pass two answers a different question.
    merged_conflicts = list(state.get("conflicts") or []) + conflicts
    record(
        "reconcile",
        f"status={parsed.status.value} conflicts={len(parsed.conflicts)} gaps={len(parsed.gaps)}",
        session_id=state["session_id"],
        node="reconcile",
        data={"reconciliation": parsed.model_dump(mode="json")},
    )

    trace_step = _trace(
        "reconcile",
        f"Reconciliation: {parsed.status.value}",
        parsed.reason,
        {
            "conflicts": parsed.conflicts,
            "gaps": parsed.gaps,
            "follow_up_queries": parsed.follow_up_queries,
        },
    )

    update: dict[str, Any] = {
        "conflicts": merged_conflicts,
        "tokens_in": state.get("tokens_in", 0) + tin,
        "tokens_out": state.get("tokens_out", 0) + tout,
        "trace": [trace_step],
    }

    if parsed.status is ReconcileStatus.ASK_USER:
        update["status"] = "needs_input"
        update["user_question"] = parsed.user_question or "Could you clarify what you need?"
    elif parsed.status is ReconcileStatus.RE_RETRIEVE:
        update["_follow_up_queries"] = parsed.follow_up_queries
    else:
        update["_follow_up_queries"] = []
    return update


def replan_from_reconcile(state: AgentState) -> dict[str, Any]:
    """Fold a reconciler's follow-up queries into the plan and re-retrieve."""
    queries = _pending_queries(state)
    if not queries or not state.get("plan"):
        return {}
    updated = append_followup_tasks(
        state["plan"],
        queries,
        why="The first pass of retrieval did not cover everything the request needs.",
        step_id="followup",
    )
    if updated == state["plan"]:
        return {}
    return {
        "plan": updated,
        "trace": [
            _trace(
                "reconcile_pass2",
                f"Re-retrieving for {len(queries)} gap(s)",
                "; ".join(queries),
            )
        ],
    }


def append_followup_tasks(
    plan_dict: dict[str, Any], queries: list[str], *, why: str, step_id: str
) -> dict[str, Any]:
    """Return ``plan_dict`` with any not-yet-planned query appended as a step.

    Used by both the reconciliation retry and the verification retry so a
    follow-up query is never dropped just because the LLM did not echo it back.
    """
    plan_obj = Plan.model_validate(plan_dict)
    existing = {t.query.strip().lower() for step in plan_obj.steps for t in step.tasks}
    added: list[RetrievalTask] = []
    for query in queries:
        if not query.strip() or query.strip().lower() in existing:
            continue
        added.append(
            RetrievalTask(
                step_id=f"{step_id}_{len(added) + 1}",
                source_type=_guess_source(query),
                query=query,
                purpose=why,
            )
        )
    if not added:
        return plan_dict
    plan_obj.steps.append(
        PlanStep(
            id=step_id,
            goal="Close the gaps identified after the first pass",
            why=why,
            sources=sorted({t.source_type for t in added}, key=lambda s: s.value),
            tasks=added,
        )
    )
    return plan_obj.model_dump(mode="json")


def _pending_queries(state: AgentState) -> list[str]:
    return list(state.get("_follow_up_queries") or [])


def _guess_source(query: str) -> SourceType:
    lowered = query.lower()
    if any(word in lowered for word in ("price", "plan", "cost", "seat", "entitlement", "storage")):
        return SourceType.CATALOG
    if any(
        word in lowered
        for word in ("policy", "sla", "window", "threshold", "approval", "entitled")
    ):
        return SourceType.POLICY
    if any(
        word in lowered
        for word in ("ticket", "customer", "subscription", "account", "how many", "list")
    ):
        return SourceType.OPS_DB
    return SourceType.KB


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #


def decide(state: AgentState, *, llm) -> dict[str, Any]:
    evidence = _load_evidence(state.get("evidence") or [])
    human = (
        f"<request>\n{state['message']}\n</request>\n\n"
        f"<intake>\n{json.dumps(state.get('intake') or {}, default=str)}\n</intake>\n\n"
        f"<plan>\n{json.dumps(state.get('plan') or {}, default=str)}\n</plan>\n\n"
        f"<evidence>\n{_render_evidence(evidence)}\n</evidence>\n\n"
        f"<reconciliation conflicts>\n"
        f"{json.dumps(state.get('conflicts') or [], default=str)}\n"
        f"</reconciliation conflicts>\n"
    )
    parsed, tin, tout = llm.structured(
        DecisionSet, system=prompts.DECIDER_PROMPT, human=human, node="decide"
    )

    accepted: list[Decision] = []
    rejected: list[str] = []
    for index, decision in enumerate(parsed.decisions):
        decision.id = decision.id or f"d{index + 1}"
        spec = get_action(decision.action)
        if spec is None:
            rejected.append(
                f"{decision.id}: '{decision.action}' is not a registered action; "
                f"available: {', '.join(sorted(get_all_action_names()))}"
            )
            continue
        args, error = validate_args(spec, decision.args)
        if error:
            rejected.append(f"{decision.id}: {error}")
            continue
        decision.args = args or {}
        decision.risk = spec.risk
        accepted.append(decision)

    for note in rejected:
        logger.warning("dropped decision %s", note)
    record(
        "decide",
        f"accepted={len(accepted)} rejected={len(rejected)} "
        f"answer_only={bool(parsed.answer_without_action)}",
        session_id=state["session_id"],
        node="decide",
        data={
            "decisions": [d.model_dump(mode="json") for d in accepted],
            "rejected": rejected,
            "insufficient_information": parsed.insufficient_information,
        },
    )

    detail = (
        f"{len(accepted)} action(s) chosen"
        if accepted
        else "no action needed; answering from evidence"
    )
    if rejected:
        detail += f"; {len(rejected)} rejected as invalid"

    return {
        "decisions": [d.model_dump(mode="json") for d in accepted],
        "tokens_in": state.get("tokens_in", 0) + tin,
        "tokens_out": state.get("tokens_out", 0) + tout,
        "trace": [
            _trace(
                "decide",
                detail,
                "; ".join(f"{d.id}:{d.action} - {d.why}" for d in accepted)
                or (parsed.answer_without_action or "")[:200],
                {
                    "decisions": [
                        {
                            "id": d.id,
                            "action": d.action,
                            "args": d.args,
                            "why": d.why,
                            "risk": d.risk.value,
                            "depends_on": d.depends_on,
                            "needs_approval": needs_approval(d.action),
                        }
                        for d in accepted
                    ],
                    "rejected": rejected,
                    "insufficient_information": parsed.insufficient_information,
                    "answer_without_action": parsed.answer_without_action,
                },
            )
        ],
    }


def get_all_action_names() -> list[str]:
    from app.tools import ACTIONS

    return list(ACTIONS)


# --------------------------------------------------------------------------- #
# approval gate
# --------------------------------------------------------------------------- #


APPROVED_STATES = {"approved", "auto_approved"}


def gate_for_approval(state: AgentState, *, settings=None) -> dict[str, Any]:
    settings = settings or get_settings()
    decisions = [Decision.model_validate(d) for d in state.get("decisions") or []]
    pending: list[ApprovalRequest] = []
    auto_records: list[ApprovalRequest] = []
    auto_approve = _auto_approve_enabled(state, settings)

    for decision in decisions:
        if needs_approval(decision.action):
            request = ApprovalRequest(
                session_id=state["session_id"],
                action=decision.action,
                args=decision.args,
                why=decision.why,
                risk=decision.risk,
                reason_for_approval=_approval_reason(decision, settings),
                evidence_refs=decision.evidence_refs,
            )
            if auto_approve:
                request.status = "auto_approved"
                request.decided_by = "auto_approve_policy"
                request.decision_reason = (
                    "Auto-approved because auto_approve_irreversible is enabled. "
                    "Set AUTO_APPROVE_IRREVERSIBLE=0 for real human sign-off."
                )
                request.decided_at = utcnow_iso()
                record(
                    "approval_auto_approved",
                    f"{request.action} ({request.risk.value})",
                    session_id=state["session_id"],
                    node="gate_for_approval",
                    data=request.model_dump(mode="json"),
                )
                try:
                    approval_store.record_auto_approval(request, settings)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("approval index write failed: %s", exc)
                auto_records.append(request)
            else:
                pending.append(request)

    if not pending:
        return {
            "approvals": [p.model_dump(mode="json") for p in auto_records],
            "trace": (
                [
                    _trace(
                        "gate_for_approval",
                        f"{len(auto_records)} action(s) auto-approved by policy",
                        "auto_approve_irreversible is enabled, so no human pause was needed",
                        {"approvals": [p.model_dump(mode="json") for p in auto_records]},
                    )
                ]
                if auto_records
                else []
            ),
        }

    record(
        "approval_requested",
        "; ".join(f"{p.action} ({p.risk.value})" for p in pending),
        session_id=state["session_id"],
        node="gate_for_approval",
        data={"approvals": [p.model_dump(mode="json") for p in pending]},
    )
    for request in pending:
        try:
            approval_store.record_request(request, settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("approval index write failed: %s", exc)

    # The gated decisions stay in ``decisions`` on purpose. after_approval is
    # what removes the ones a human rejected, so an approved action is still
    # waiting here when the run resumes.
    return {
        "approvals": [p.model_dump(mode="json") for p in pending],
        "status": "awaiting_approval",
        "trace": [
            _trace(
                "gate_for_approval",
                f"{len(pending)} action(s) need human approval",
                "; ".join(f"{p.action}: {p.reason_for_approval}" for p in pending),
                {"approvals": [p.model_dump(mode="json") for p in pending]},
            )
        ],
    }


def _auto_approve_enabled(state: AgentState, settings) -> bool:
    override = state.get("auto_approve")
    if override is not None:
        return bool(override)
    return bool(getattr(settings, "auto_approve_irreversible", False))


def _approval_reason(decision: Decision, settings=None) -> str:
    settings = settings or get_settings()
    try:
        rule = approval_rule()
    except Exception:  # noqa: BLE001
        rule = {}
    threshold = rule.get("auto_execute_up_to_usd")
    if decision.action == "apply_account_credit" and threshold is not None:
        amount = (decision.args or {}).get("amount_usd")
        return (
            f"Credits move money and are irreversible. The policy allows automatic execution "
            f"up to {threshold} USD"
            + (f", and this request is for {amount} USD" if amount is not None else "")
            + ", so a human must authorise it."
        )
    return (
        f"'{decision.action}' is marked irreversible in the action registry, so a human must "
        f"authorise it before it runs."
    )


def collect_approval(state: AgentState) -> dict[str, Any]:
    """Pause the run until a human decides.

    Nothing is written to state before the interrupt, because LangGraph discards
    a node's writes when it interrupts. The pending approvals are already in
    state, written by ``gate_for_approval``.
    """
    pending = [a for a in (state.get("approvals") or []) if a.get("status") == "pending"]
    if not pending:
        return {"trace": [_trace("collect_approval", "No approval outstanding")]}

    response = interrupt(
        {
            "type": "approval_required",
            "session_id": state["session_id"],
            "message": "These actions need a human decision before they run.",
            "approvals": pending,
        }
    )

    decisions = _normalise_resume(response, pending)
    updated: list[dict[str, Any]] = []
    for approval in state.get("approvals") or []:
        match = next((d for d in decisions if d["action"] == approval["action"]), None)
        if approval.get("status") == "pending" and match is not None:
            approval["status"] = "approved" if match["approved"] else "rejected"
            approval["decided_by"] = match.get("approver", "unknown")
            approval["decision_reason"] = match.get("reason", "")
            approval["approver_role"] = match.get("role", "") or ""
            approval["decided_at"] = utcnow_iso()
            try:
                approval_store.record_decision(
                    approval["id"],
                    approved=match["approved"],
                    approver=match.get("approver", "unknown"),
                    reason=match.get("reason", ""),
                    decided_at=approval["decided_at"],
                    approver_role=approval["approver_role"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("approval index update failed: %s", exc)
        updated.append(approval)

    approved_actions = [d["action"] for d in decisions if d["approved"]]
    rejected_actions = [d["action"] for d in decisions if not d["approved"]]

    for decision in updated:
        record(
            "approval_decided",
            f"{decision['action']} -> {decision['status']} by {decision.get('decided_by')}",
            session_id=state["session_id"],
            node="collect_approval",
            data=decision,
        )

    return {
        "approvals": updated,
        "status": "running",
        "trace": [
            _trace(
                "collect_approval",
                f"Human decided {len(decisions)} request(s)",
                f"approved: {approved_actions or 'none'}; rejected: {rejected_actions or 'none'}",
                {"decisions": decisions},
            )
        ],
    }


def _normalise_resume(
    response: Any, pending: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if response is None:
        return [
            {
                "action": a["action"],
                "approved": False,
                "approver": "system",
                "role": "",
                "reason": "no response",
            }
            for a in pending
        ]
    if isinstance(response, bool):
        return [
            {
                "action": a["action"],
                "approved": response,
                "approver": "human",
                "role": "",
                "reason": "bulk decision",
            }
            for a in pending
        ]
    if isinstance(response, dict) and "decisions" in response:
        items = response["decisions"]
    elif isinstance(response, dict):
        items = [response]
    elif isinstance(response, list):
        items = response
    else:
        items = []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        approved = bool(item.get("approved", item.get("status") == "approved"))
    out.append(
        {
            "action": item.get("action", ""),
            "approved": approved,
            "approver": item.get("approver") or item.get("decided_by") or "human",
            # Carried through so policy role checks (e.g. a refund above the
            # ceiling) see the approver's role rather than an empty string.
            "role": item.get("role", "") or "",
            "reason": item.get("reason", ""),
        }
    )
    if not out:
        out = [
            {
                "action": a["action"],
                "approved": False,
                "approver": "system",
                "role": "",
                "reason": "empty decision",
            }
            for a in pending
        ]
    return out


def after_approval(state: AgentState) -> dict[str, Any]:
    """Drop decisions a human rejected and record them as skipped outcomes."""
    decisions = [Decision.model_validate(d) for d in state.get("decisions") or []]
    approved_actions = {
        a["action"] for a in (state.get("approvals") or []) if a.get("status") in APPROVED_STATES
    }
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for decision in decisions:
        if needs_approval(decision.action) and decision.action not in approved_actions:
            skipped.append(decision.model_dump(mode="json"))
        else:
            kept.append(decision.model_dump(mode="json"))
    if not skipped:
        return {"decisions": kept}
    return {
        "decisions": kept,
        "actions": [
            ActionOutcome(
                action=d["action"],
                ok=False,
                result={},
                error="rejected by human approver",
                executed_by="skipped",
                why=d["why"],
            ).model_dump(mode="json")
            for d in skipped
        ],
        "trace": [
            _trace(
                "after_approval",
                f"{len(skipped)} action(s) dropped after rejection",
                "; ".join(d["action"] for d in skipped),
            )
        ],
    }


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #


def execute(state: AgentState, *, settings) -> dict[str, Any]:
    decisions = [Decision.model_validate(d) for d in state.get("decisions") or []]
    if not decisions:
        return {
            "trace": [_trace("execute", "No actions to run", "the decision was to answer only")]
        }

    approval_by_action = {
        a["action"]: a
        for a in (state.get("approvals") or [])
        if a.get("status") in APPROVED_STATES
    }

    # Record intent before running anything, so a crash mid-execution still
    # leaves a record of what the agent was about to do.
    for decision in decisions:
        if get_action(decision.action) is None:
            continue
        record(
            "action_start",
            f"{decision.action} {json.dumps(decision.args, default=str)[:200]}",
            session_id=state["session_id"],
            node="execute",
            data={
                "decision": decision.model_dump(mode="json"),
                "evidence_refs": decision.evidence_refs,
                "approval_id": (approval_by_action.get(decision.action) or {}).get("id"),
            },
        )

    def make_call(decision: Decision) -> Callable[[], ActionOutcome]:
        def call() -> ActionOutcome:
            spec = get_action(decision.action)
            if spec is None:
                return ActionOutcome(
                    action=decision.action,
                    ok=False,
                    error="action not registered",
                    why=decision.why,
                )
            approval = approval_by_action.get(decision.action)
            ctx = ActionContext(
                session_id=state["session_id"],
                actor=approval.get("decided_by", "agent") if approval else "agent",
                why=decision.why,
                evidence_refs=decision.evidence_refs,
                approval_id=approval.get("id") if approval else None,
                user_id=state.get("user_id", "anonymous"),
                approver_role=(approval or {}).get("approver_role", ""),
            )
            started = time.perf_counter()
            try:
                result = spec.fn(decision.args, ctx)
            except Exception as exc:  # noqa: BLE001
                outcome = ActionOutcome(
                    action=decision.action,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    executed_by="human" if approval else "agent",
                    approval_id=ctx.approval_id,
                    why=decision.why,
                )
                record(
                    "action_failed",
                    f"{decision.action}: {outcome.error}",
                    session_id=state["session_id"],
                    node="execute",
                    data={"decision": decision.model_dump(mode="json"), "error": outcome.error},
                )
                return outcome
            outcome = ActionOutcome(
                action=decision.action,
                ok=True,
                result=result,
                duration_ms=int((time.perf_counter() - started) * 1000),
                executed_by="human" if approval else "agent",
                approval_id=ctx.approval_id,
                why=decision.why,
            )
            record(
                "action_succeeded",
                f"{decision.action} -> {json.dumps(result, default=str)[:200]}",
                session_id=state["session_id"],
                node="execute",
                data={"decision": decision.model_dump(mode="json"), "result": result},
            )
            return outcome

        return call

    ordered = _dependency_order(decisions)
    outcomes: list[ActionOutcome] = []
    for tier in ordered:
        results = _run_tasks_in_parallel([make_call(d) for d in tier])
        for result in results:
            if isinstance(result, ActionOutcome):
                outcomes.append(result)
            else:
                outcomes.append(
                    ActionOutcome(
                        action="unknown",
                        ok=False,
                        error=f"{type(result).__name__}: {result}",
                    )
                )

    ok_count = sum(1 for o in outcomes if o.ok)
    # Accumulate across iterations: these are the actions that actually happened,
    # and a retry must not erase the record of what the first pass wrote.
    prior_actions = [
        ActionOutcome.model_validate(a)
        for a in (state.get("actions") or [])
        if a.get("action") not in {o.action for o in outcomes}
    ]
    all_outcomes = prior_actions + outcomes
    return {
        "actions": [o.model_dump(mode="json") for o in all_outcomes],
        "trace": [
            _trace(
                "execute",
                f"Executed {len(outcomes)} action(s), {ok_count} succeeded",
                "; ".join(f"{o.action}:{'ok' if o.ok else 'FAILED'}" for o in outcomes),
                {
                    "outcomes": [
                        {
                            "action": o.action,
                            "ok": o.ok,
                            "result": o.result,
                            "error": o.error,
                            "executed_by": o.executed_by,
                        }
                        for o in outcomes
                    ],
                    "parallel_tiers": [[d.action for d in tier] for tier in ordered],
                },
            )
        ],
    }


def _dependency_order(decisions: list[Decision]) -> list[list[Decision]]:
    """Group decisions into tiers so dependents run after what they depend on."""
    by_ref: dict[str, Decision] = {}
    for decision in decisions:
        by_ref[decision.id] = decision
        by_ref.setdefault(decision.action, decision)
    placed: set[str] = set()
    tiers: list[list[Decision]] = []
    remaining = list(decisions)
    while remaining:
        tier = [
            d
            for d in remaining
            if all(
                (ref in placed) or (ref not in by_ref) or (by_ref[ref] is d)
                for ref in d.depends_on
            )
        ]
        if not tier:
            tier = remaining[:]
        for decision in tier:
            placed.add(decision.id)
            placed.add(decision.action)
        remaining = [d for d in remaining if d not in tier]
        tiers.append(tier)
    return tiers


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def verify(state: AgentState, *, llm) -> dict[str, Any]:
    outcomes = state.get("actions") or []
    actions_text = json.dumps(outcomes, default=str)[:4000]
    human = (
        f"<goal>\n{(state.get('plan') or {}).get('goal', state['message'])}\n</goal>\n\n"
        f"<request>\n{state['message']}\n</request>\n\n"
        f"<actions attempted and their results>\n{actions_text}\n"
        f"</actions attempted and their results>\n\n"
        f"<conflicts found>\n{json.dumps(state.get('conflicts') or [], default=str)}\n"
        f"</conflicts found>\n"
    )
    parsed, tin, tout = llm.structured(
        Verification, system=prompts.VERIFIER_PROMPT, human=human, node="verify"
    )
    record(
        "verify",
        f"achieved={parsed.achieved} confidence={parsed.confidence:.2f}",
        session_id=state["session_id"],
        node="verify",
        data={"verification": parsed.model_dump(mode="json")},
    )
    return {
        "verification": parsed.model_dump(mode="json"),
        "tokens_in": state.get("tokens_in", 0) + tin,
        "tokens_out": state.get("tokens_out", 0) + tout,
        "trace": [
            _trace(
                "verify",
                f"Verification {'passed' if parsed.achieved else 'incomplete'} "
                f"(confidence {parsed.confidence:.2f})",
                parsed.reasoning,
                {"missing": parsed.missing, "follow_up_queries": parsed.follow_up_queries},
            )
        ],
    }


def after_verify(state: AgentState) -> dict[str, Any]:
    iteration = state.get("iteration", 0) + 1
    update: dict[str, Any] = {"iteration": iteration}
    verification = state.get("verification") or {}
    if iteration < state.get("max_iterations", 3) and not verification.get("achieved", True):
        queries = verification.get("follow_up_queries") or []
        if queries:
            update["_follow_up_queries"] = queries
    return update


# --------------------------------------------------------------------------- #
# response
# --------------------------------------------------------------------------- #


def _stream_writer():
    """The LangGraph custom-stream sink, or None when not streaming.

    ``get_stream_writer`` only exists meaningfully inside a run that asked for
    ``stream_mode="custom"``. Under a plain ``invoke`` there is no sink, and the
    node must fall back to a single blocking call so behaviour is identical
    either way.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001
        return None


def _generate_answer(llm, *, system: str, human: str) -> tuple[str, int, int]:
    """Produce the final answer, streaming it when the caller is listening.

    Emits ``{"type": "token", "text": ...}`` per delta so the API can forward
    progress over SSE. The accumulated text is identical to the non-streaming
    path, and the state update is unchanged.
    """
    writer = _stream_writer()
    if writer is None or not hasattr(llm, "stream_text"):
        return llm.text(system=system, human=human, node="respond")

    usage: dict[str, int] = {}
    parts: list[str] = []
    for delta in llm.stream_text(system=system, human=human, node="respond", usage=usage):
        if not delta:
            continue
        parts.append(delta)
        try:
            writer({"type": "token", "text": delta})
        except Exception as exc:  # noqa: BLE001
            # A broken sink must not lose the answer.
            logger.warning("stream writer failed (%s); continuing", exc)
    # Deltas are joined verbatim, so a provider that splits on whitespace leaves
    # a trailing space the non-streaming path would not have. Strip it so the
    # two paths return byte-identical answers.
    return "".join(parts).strip(), usage.get("in", 0), usage.get("out", 0)


def respond(state: AgentState, *, llm, memory) -> dict[str, Any]:
    evidence = _load_evidence(state.get("evidence") or [])
    actions = state.get("actions") or []
    decisions = state.get("decisions") or []
    verification = state.get("verification") or {}
    approvals = state.get("approvals") or []

    human = (
        f"<request>\n{state['message']}\n</request>\n\n"
        f"<intake>\n{json.dumps(state.get('intake') or {}, default=str)}\n</intake>\n\n"
        f"<plan>\n{json.dumps(state.get('plan') or {}, default=str)}\n</plan>\n\n"
        f"<conflicts>\n{json.dumps(state.get('conflicts') or [], default=str)}\n</conflicts>\n\n"
        f"<decisions and whether they were executed>\n"
        f"{json.dumps({'decisions': decisions, 'outcomes': actions}, default=str)[:4000]}\n"
        f"</decisions and whether they were executed>\n\n"
        f"<approvals>\n{json.dumps(approvals, default=str)[:2500]}\n</approvals>\n\n"
        f"<verification>\n{json.dumps(verification, default=str)}\n</verification>\n\n"
        f"<evidence, cite by citation label>\n{_render_evidence(evidence, limit=30)}\n"
        f"</evidence, cite by citation label>\n"
    )
    answer, tin, tout = _generate_answer(llm, system=prompts.RESPONDER_PROMPT, human=human)

    facts: dict[str, Any] = {
        "last_message": state["message"][:400],
        "intent": (state.get("intake") or {}).get("intent"),
        "customer_id": (state.get("intake") or {}).get("customer_id", ""),
        "ticket_id": (state.get("intake") or {}).get("ticket_id", ""),
        "actions_run": [
            {"action": o.get("action"), "ok": o.get("ok")} for o in actions
        ],
        "achieved": verification.get("achieved"),
        "answered_at": utcnow_iso(),
    }
    try:
        memory.remember("turn", "latest_turn", facts, session_id=state["session_id"])
        if facts["customer_id"]:
            memory.remember(
                "interaction", "last_interaction", facts, user_id=facts["customer_id"]
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory write failed: %s", exc)

    record(
        "respond",
        f"answer_chars={len(answer)} status={state.get('status')}",
        session_id=state["session_id"],
        node="respond",
        data={"answer": answer},
    )

    return {
        "answer": answer,
        "status": state.get("status") or "completed",
        "tokens_in": state.get("tokens_in", 0) + tin,
        "tokens_out": state.get("tokens_out", 0) + tout,
        "trace": [
            _trace(
                "respond",
                "Wrote the final answer",
                f"{len(answer)} characters",
                {"iterations": state.get("iteration", 0)},
            )
        ],
    }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class SqlQuery(BaseModel):
    sql: str = Field(description="A single read-only SELECT statement.")


SQL_WRITER_PROMPT = """You write one read-only SELECT statement against the operations
database. Given a question, produce the query that answers it.

Rules:
- One statement only, no semicolons, no comments.
- SELECT only. No INSERT, UPDATE, DELETE, DROP, ALTER or PRAGMA.
- Include an explicit LIMIT.
- Only reference tables and columns that exist in the schema you were given.
- Return empty results rather than guessing at a column that is not in the schema.
"""
