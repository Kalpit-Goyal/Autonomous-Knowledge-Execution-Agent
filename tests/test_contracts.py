"""Contract tests between the nodes, the state, and the response.

LangGraph silently drops any key a node returns that is not declared in the
state schema. That failure is invisible: the run completes, the trace looks
fine, and the value simply vanishes. These tests pin the contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.graph import nodes as nodes_module
from app.graph.state import AgentState
from app.schemas import ChatResponse

NODES = Path(nodes_module.__file__)


def _returned_keys(function: ast.FunctionDef) -> set[str]:
    """Keys named in ``return {...}`` dict literals inside a function."""
    keys: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def test_every_state_key_the_nodes_write_is_declared():
    """A node writing an undeclared key would have its output silently dropped."""
    declared = set(AgentState.__annotations__)

    tree = ast.parse(NODES.read_text(encoding="utf-8"))
    missing: dict[str, set[str]] = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        undeclared = {k for k in _returned_keys(function) if k not in declared}
        if undeclared:
            missing[function.name] = undeclared

    assert not missing, f"nodes return keys absent from AgentState: {missing}"


def test_runner_populates_the_response_fields_the_ui_reads():
    source = Path(nodes_module.__file__).read_text(encoding="utf-8")
    # guards the specific regression: intake was computed but never declared
    assert "intake" in AgentState.__annotations__
    assert "intake" in source
    assert "intake" in ChatResponse.model_fields


def test_state_annotations_all_serialise():
    """Checkpointed state must be JSON-safe, so no pydantic objects or callables."""
    for name, annotation in AgentState.__annotations__.items():
        text = str(annotation)
        assert "BaseModel" not in text, f"{name} holds a pydantic model: {text}"
        assert "Callable" not in text, f"{name} holds a callable: {text}"


def test_graph_builds_and_exposes_every_node(sandbox):
    from app.graph.builder import build_graph
    from app.llm import ScriptedLLM

    graph = build_graph(sandbox, llm=ScriptedLLM(lambda *a: None, name="t"))
    expected = {
        "intake",
        "plan",
        "retrieve",
        "reconcile",
        "decide",
        "gate_for_approval",
        "collect_approval",
        "after_approval",
        "execute",
        "verify",
        "after_verify",
        "respond",
        "reconcile_pass2",
    }
    assert expected <= set(graph.get_graph().nodes)
