"""Graph assembly.

    intake -> plan -> retrieve -> reconcile -> decide -> gate -> execute
                   ^                |                     |        |
                   |                v                     v        v
                   +--- replan ----+                 (approve)  verify
                                                                        |
                            respond <--- loop while unverified <--------+

Two conditional edges carry most of the control flow:

* ``reconcile`` may send the run back to ``retrieve`` for a second pass, or
  straight to ``respond`` when only the requester can answer.
* ``verify`` sends the run back to ``plan`` while the goal is unproven and the
  iteration budget allows it.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.graph import nodes
from app.graph.state import AgentState
from app.llm import build_llm
from app.memory import get_memory_store

logger = logging.getLogger(__name__)

_saver: SqliteSaver | None = None
_saver_conn: sqlite3.Connection | None = None


def get_checkpointer(settings: Settings | None = None) -> SqliteSaver:
    """One long-lived SQLite checkpointer.

    Durable on purpose: an approval request must still be waiting after a
    restart, otherwise a pause would silently lose a pending human decision.
    """
    global _saver, _saver_conn
    if _saver is not None:
        return _saver
    settings = settings or get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    _saver_conn = sqlite3.connect(
        settings.checkpoint_db_path, timeout=15, check_same_thread=False
    )
    _saver_conn.row_factory = sqlite3.Row
    _saver = SqliteSaver(_saver_conn)
    _saver.setup()
    return _saver


def build_graph(settings: Settings | None = None, *, llm=None, memory=None):
    settings = settings or get_settings()
    llm = llm if llm is not None else build_llm(settings)
    memory = memory if memory is not None else get_memory_store(settings)

    graph = StateGraph(AgentState)

    graph.add_node("intake", lambda s: nodes.intake(s, llm=llm, memory=memory))
    graph.add_node("plan", lambda s: nodes.plan(s, llm=llm))
    graph.add_node("retrieve", lambda s: nodes.retrieve(s, llm=llm))
    graph.add_node("reconcile", lambda s: nodes.reconcile(s, llm=llm))
    graph.add_node("decide", lambda s: nodes.decide(s, llm=llm))
    graph.add_node("gate_for_approval", lambda s: nodes.gate_for_approval(s, settings=settings))
    graph.add_node("collect_approval", nodes.collect_approval)
    graph.add_node("after_approval", nodes.after_approval)
    graph.add_node("execute", lambda s: nodes.execute(s, settings=settings))
    graph.add_node("verify", lambda s: nodes.verify(s, llm=llm))
    graph.add_node("after_verify", nodes.after_verify)
    graph.add_node("respond", lambda s: nodes.respond(s, llm=llm, memory=memory))

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "reconcile")

    # a plain pass-through node so re-retrieval keeps a distinct trace entry
    graph.add_node("reconcile_pass2", nodes.replan_from_reconcile)

    graph.add_conditional_edges(
        "reconcile",
        _route_after_reconcile,
        {
            "re_retrieve": "reconcile_pass2",
            "ask_user": "respond",
            "proceed": "decide",
        },
    )
    graph.add_edge("reconcile_pass2", "retrieve")

    graph.add_edge("decide", "gate_for_approval")

    graph.add_conditional_edges(
        "gate_for_approval",
        _route_after_gate,
        {"await": "collect_approval", "run": "execute"},
    )
    graph.add_edge("collect_approval", "after_approval")
    graph.add_edge("after_approval", "execute")
    graph.add_edge("execute", "verify")

    # verify -> after_verify is unconditional: after_verify owns the iteration
    # counter, so the budget is spent even on the run that succeeds.
    graph.add_edge("verify", "after_verify")
    graph.add_conditional_edges(
        "after_verify",
        _route_after_verify,
        {"retry": "plan", "done": "respond"},
    )

    graph.add_edge("respond", END)

    return graph.compile(checkpointer=get_checkpointer(settings))


def _route_after_reconcile(state: AgentState) -> str:
    if state.get("status") == "needs_input":
        return "ask_user"
    if state.get("_follow_up_queries"):
        return "re_retrieve"
    return "proceed"


def _route_after_gate(state: AgentState) -> str:
    pending = [a for a in (state.get("approvals") or []) if a.get("status") == "pending"]
    return "await" if pending else "run"


def _route_after_verify(state: AgentState) -> str:
    verification = state.get("verification") or {}
    if verification.get("achieved", True):
        return "done"
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        return "done"
    if verification.get("follow_up_queries"):
        return "retry"
    return "done"


_compiled: dict[str, Any] = {}


def get_compiled_graph(settings: Settings | None = None, *, llm=None, memory=None):
    """Cache the compiled graph so the FastAPI process builds it once."""
    settings = settings or get_settings()
    key = f"{settings.data_dir}|{id(llm)}|{id(memory)}"
    if key not in _compiled:
        _compiled[key] = build_graph(settings, llm=llm, memory=memory)
    return _compiled[key]
