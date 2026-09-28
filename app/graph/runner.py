"""Invoking the graph, including the approval pause.

``run`` handles the awkward part of ``interrupt``: a paused run raises
``GraphInterrupt`` rather than returning a result, and the caller has to be able
to tell "waiting for a human" apart from "crashed". Both cases are normalised
here so the API and the Streamlit client only ever see a ``ChatResponse``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

from langgraph.types import Command

from app.config import Settings, get_settings
from app.graph.builder import get_compiled_graph
from app.schemas import (
    ActionOutcome,
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    Decision,
    Evidence,
    Intake,
    Plan,
    TraceStep,
    Verification,
)

logger = logging.getLogger(__name__)

INTERRUPT_MARKER = "__interrupt__"


def new_session_id() -> str:
    return f"SESS-{uuid.uuid4().hex[:10]}"


def _initial_state(request: ChatRequest, session_id: str, settings: Settings) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "user_id": request.user_id,
        "message": request.message,
        "auto_approve": request.auto_approve,
        "iteration": 0,
        "max_iterations": settings.max_iterations,
        "plan": None,
        "evidence": [],
        "conflicts": [],
        "decisions": [],
        "actions": [],
        "approvals": [],
        "trace": [],
        "answer": "",
        "status": "running",
        "user_question": None,
        "verification": None,
        "retrieved_tool_calls": [],
        "memory_hits": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "error": None,
        "_follow_up_queries": [],
    }


def stream_run(
    request: ChatRequest,
    *,
    settings: Settings | None = None,
    graph=None,
    llm=None,
    memory=None,
) -> Iterator[dict[str, Any]]:
    """Run the graph, yielding progress events as they happen.

    Emits ``start``, then ``node`` for each node transition, then ``token`` for
    each answer delta, and finally exactly one of ``done`` or ``error``. The
    final ``done`` payload is the same ``ChatResponse`` the blocking ``run``
    returns, so a streamed and a non-streamed turn are interchangeable.

    The graph is driven with ``stream_mode=["custom", "updates"]``: ``custom``
    carries the responder's deltas (written by the node through
    ``get_stream_writer``) and ``updates`` carries per-node state so the client
    can show progress while the reasoning nodes run.
    """
    settings = settings or get_settings()
    graph = graph or get_compiled_graph(settings, llm=llm, memory=memory)
    session_id = request.session_id or new_session_id()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 60,
    }
    state = _initial_state(request, session_id, settings)

    started = time.perf_counter()
    yield {"event": "start", "session_id": session_id}

    interrupted = False
    try:
        for mode, chunk in graph.stream(
            state, config=config, stream_mode=["custom", "updates"]
        ):
            if mode == "custom":
                for item in _iter_tokens(chunk):
                    yield {"event": "token", "text": item}
                continue
            for node, _update in _iter_updates(chunk):
                if node == INTERRUPT_MARKER:
                    interrupted = True
                    continue
                yield {"event": "node", "node": node}
    except Exception as exc:  # noqa: BLE001
        logger.exception("graph stream failed")
        yield {
            "event": "error",
            "session_id": session_id,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
        return

    payload = _read_checkpoint(graph, config, session_id)
    if interrupted or payload.get(INTERRUPT_MARKER) or payload.get("__interrupt__"):
        payload.pop(INTERRUPT_MARKER, None)
        payload.pop("__interrupt__", None)
        payload.setdefault("status", "awaiting_approval")
    payload.setdefault("status", "completed")

    response = _to_response(payload, session_id, started, settings)
    yield {
        "event": "done",
        "session_id": response.session_id,
        "response": response.model_dump(mode="json"),
    }


def _iter_tokens(chunk: Any) -> Iterator[str]:
    """Pull ``{"type": "token", "text": ...}`` writes out of a custom chunk."""
    if isinstance(chunk, dict):
        candidates: list[Any] = [chunk]
    elif isinstance(chunk, (list, tuple)):
        candidates = list(chunk)
    else:
        return
    for item in candidates:
        if isinstance(item, dict) and item.get("type") == "token" and item.get("text"):
            yield str(item["text"])


def _iter_updates(chunk: Any) -> Iterator[tuple[str, Any]]:
    if not isinstance(chunk, dict):
        return
    for node, update in chunk.items():
        if node in {INTERRUPT_MARKER, "__interrupt__"} or (
            isinstance(update, (list, tuple)) and update
        ):
            yield INTERRUPT_MARKER, update
        else:
            yield node, update


def run(
    request: ChatRequest,
    *,
    settings: Settings | None = None,
    graph=None,
    llm=None,
    memory=None,
) -> ChatResponse:
    settings = settings or get_settings()
    graph = graph or get_compiled_graph(settings, llm=llm, memory=memory)
    session_id = request.session_id or new_session_id()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 60,
    }

    started = time.perf_counter()
    state = _initial_state(request, session_id, settings)
    try:
        result = graph.invoke(state, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("graph run failed")
        return ChatResponse(
            session_id=session_id,
            answer="",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    payload = dict(result or {})
    if payload.pop(INTERRUPT_MARKER, None) or _is_interrupted(result):
        payload.update(_read_checkpoint(graph, config, session_id))
        payload.setdefault("status", "awaiting_approval")

    return _to_response(payload, session_id, started, settings)


def resume(
    session_id: str,
    decisions: list[dict[str, Any]] | dict[str, Any],
    *,
    settings: Settings | None = None,
    graph=None,
    llm=None,
    memory=None,
) -> ChatResponse:
    settings = settings or get_settings()
    graph = graph or get_compiled_graph(settings, llm=llm, memory=memory)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 60,
    }
    resume_value = (
        decisions if isinstance(decisions, dict) else {"decisions": decisions}
    )
    started = time.perf_counter()
    try:
        result = graph.invoke(Command(resume=resume_value), config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("graph resume failed")
        return ChatResponse(
            session_id=session_id,
            answer="",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    payload = dict(result or {})
    if _is_interrupted(result):
        payload.update(_read_checkpoint(graph, config, session_id))
        payload.setdefault("status", "awaiting_approval")
    return _to_response(payload, session_id, started, settings)


def _is_interrupted(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get(INTERRUPT_MARKER):
        return True
    interrupts = result.get("__interrupt__")
    return bool(interrupts)


def _read_checkpoint(graph, config: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Pull the state saved at the moment of the pause."""
    snapshot = graph.get_state(config)
    values = dict(getattr(snapshot, "values", None) or {})
    pending = []
    for task in getattr(snapshot, "tasks", None) or []:
        for item in getattr(task, "interrupts", None) or []:
            value = getattr(item, "value", None)
            if isinstance(value, dict) and value.get("type") == "approval_required":
                pending.extend(value.get("approvals") or [])
    if pending:
        existing = {a.get("id") for a in values.get("approvals") or []}
        merged = list(values.get("approvals") or [])
        for item in pending:
            if item.get("id") not in existing:
                merged.append(item)
        values["approvals"] = merged
    values["session_id"] = values.get("session_id") or session_id
    return values


def _to_response(
    payload: dict[str, Any], session_id: str, started: float, settings: Settings
) -> ChatResponse:
    status = payload.get("status") or "completed"
    if status not in {"completed", "awaiting_approval", "needs_input", "error"}:
        status = "completed"
    approvals = [
        ApprovalRequest.model_validate(a) for a in payload.get("approvals") or []
    ]
    pending = [a for a in approvals if a.status == "pending"]

    answer = payload.get("answer") or ""
    if not answer and status == "awaiting_approval":
        answer = _approval_hold_message(pending)
    elif not answer and status == "needs_input":
        answer = payload.get("user_question") or "I need one more detail before I can continue."

    return ChatResponse(
        session_id=payload.get("session_id") or session_id,
        answer=answer,
        status=status,
        citations=[Evidence.model_validate(e) for e in payload.get("evidence") or []],
        conflicts=list(payload.get("conflicts") or []),
        decisions=[
            Decision.model_validate(d) for d in payload.get("decisions") or []
        ],
        actions=[ActionOutcome.model_validate(a) for a in payload.get("actions") or []],
        approvals=approvals,
        trace=[TraceStep.model_validate(t) for t in payload.get("trace") or []],
        plan=Plan.model_validate(payload["plan"]) if payload.get("plan") else None,
        intake=Intake.model_validate(payload["intake"]) if payload.get("intake") else None,
        verification=(
            Verification.model_validate(payload["verification"])
            if payload.get("verification")
            else None
        ),
        user_question=payload.get("user_question"),
        iterations=int(payload.get("iteration") or 0),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        error=payload.get("error"),
    )


def _approval_hold_message(pending: list[ApprovalRequest]) -> str:
    if not pending:
        return "Waiting on a human decision."
    lines = ["I have stopped short of changing anything. These actions need a human decision:", ""]
    for item in pending:
        lines.append(f"- {item.action}: {item.reason_for_approval}")
        lines.append(f"  why: {item.why}")
    lines.append("")
    lines.append("Approve or reject them in the approval panel and I will continue from here.")
    return "\n".join(lines)
