"""Graph state.

Everything here must survive JSON serialisation because it is checkpointed -
that is what lets an approval pause, a process restart, and a human resume the
same run. Structured LLM output is therefore stored as plain dicts and
revalidated on the way out rather than held as pydantic objects.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


def merge_lists(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Append-right with de-duplication.

    LangGraph reducers run whenever two branches write the same key, so this
    keeps parallel retrieval branches from clobbering each other.
    """
    merged: list[Any] = list(left or [])
    for item in right or []:
        if item not in merged:
            merged.append(item)
    return merged


def replace_or_append(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Replace on a later write, append during a single fan-in.

    Retrieval and execution both fan out across several calls at once, so the
    first concurrent write should accumulate; once the node completes it writes
    the full list again, and that write must win outright.
    """
    return list(right or [])


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    message: str
    auto_approve: bool | None

    iteration: int
    max_iterations: int
    intake: dict[str, Any] | None
    plan: dict[str, Any] | None
    evidence: Annotated[list[dict[str, Any]], replace_or_append]
    conflicts: Annotated[list[dict[str, Any]], replace_or_append]
    decisions: Annotated[list[dict[str, Any]], replace_or_append]
    actions: Annotated[list[dict[str, Any]], replace_or_append]
    approvals: Annotated[list[dict[str, Any]], replace_or_append]
    trace: Annotated[list[dict[str, Any]], merge_lists]
    messages: Annotated[list[Any], add_messages]

    answer: str
    status: str
    user_question: str | None
    verification: dict[str, Any] | None
    retrieved_tool_calls: list[dict[str, Any]]
    memory_hits: list[dict[str, Any]]
    tokens_in: int
    tokens_out: int
    error: str | None
    _follow_up_queries: list[str]


def initial_state(
    session_id: str,
    user_id: str,
    message: str,
    max_iterations: int,
    auto_approve: bool | None = None,
) -> AgentState:
    return {
        "session_id": session_id,
        "user_id": user_id,
        "message": message,
        "auto_approve": auto_approve,
        "iteration": 0,
        "max_iterations": max_iterations,
        "intake": None,
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
    }
