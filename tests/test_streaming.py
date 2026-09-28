"""Streaming responses.

The graph must produce the *same* answer whether it is driven by ``invoke`` or
by ``stream``; only the delivery differs. These tests pin both halves of that:
the deltas concatenate back to the blocking answer, and the final ``done`` event
carries a ``ChatResponse`` indistinguishable from the blocking one.

The responder here is scripted, so the stream is real (one event per word) with
no provider and no quota.
"""

from __future__ import annotations

import pytest

from app.graph.runner import run, stream_run
from app.llm import ScriptedLLM
from app.memory import get_memory_store
from app.schemas import ChatRequest

ANSWER = "Streamed answer with a citation."
QUESTION = "What is the refund window for an annual plan?"


def _build(sandbox, answer: str = ANSWER):
    from app.graph.builder import build_graph
    from tests.test_graph import make_responder

    llm = ScriptedLLM(
        make_responder(),
        lambda system, human, node: answer,
        name="test",
    )
    return build_graph(sandbox, llm=llm, memory=get_memory_store(sandbox))


def _events(sandbox, message: str = QUESTION, answer: str = ANSWER) -> list[dict]:
    graph = _build(sandbox, answer)
    return list(
        stream_run(
            ChatRequest(message=message),
            settings=sandbox,
            graph=graph,
        )
    )


def test_stream_emits_start_then_tokens_then_a_single_done(sandbox):
    events = _events(sandbox)
    kinds = [e["event"] for e in events]

    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert kinds.count("done") == 1
    assert "error" not in kinds
    assert "token" in kinds, "the responder should have streamed at least one token"


def test_tokens_reconstruct_exactly_the_blocking_answer(sandbox):
    events = _events(sandbox)
    streamed = "".join(e["text"] for e in events if e["event"] == "token").strip()
    assert streamed == ANSWER

    # The blocking path must agree, or a client that toggles streaming on and off
    # would see different answers for the same question.
    graph = _build(sandbox)
    blocked = run(ChatRequest(message=QUESTION), settings=sandbox, graph=graph)
    assert blocked.answer.strip() == streamed


def test_done_event_carries_a_full_chat_response(sandbox):
    events = _events(sandbox)
    done = [e for e in events if e["event"] == "done"][0]
    response = done["response"]

    assert response["status"] == "completed"
    assert response["answer"].strip() == ANSWER
    assert response["session_id"].startswith("SESS-")
    assert response["elapsed_ms"] >= 0
    # The structured fields the UI renders must all survive the streaming path.
    assert isinstance(response["trace"], list) and response["trace"]
    assert isinstance(response["citations"], list)
    assert response["plan"] is not None
    assert response["intake"] is not None


def test_node_events_cover_the_reasoning_path(sandbox):
    nodes = [e["node"] for e in _events(sandbox) if e["event"] == "node"]
    for expected in ("intake", "plan", "retrieve", "reconcile", "decide", "respond"):
        assert expected in nodes, f"{expected} did not report progress"


def test_stream_reports_the_same_session_id_it_started_with(sandbox):
    events = _events(sandbox)
    start = [e for e in events if e["event"] == "start"][0]
    done = [e for e in events if e["event"] == "done"][0]
    assert start["session_id"] == done["session_id"]


def test_streaming_pause_reports_awaiting_approval(sandbox):
    """An interrupt must end as a normal ``done`` with a hold, not an error."""
    from app.graph.builder import build_graph
    from tests.test_graph import _credit_decision_set, make_responder

    llm = ScriptedLLM(
        make_responder(decide=_credit_decision_set()),
        lambda system, human, node: ANSWER,
        name="test",
    )
    graph = build_graph(sandbox, llm=llm, memory=get_memory_store(sandbox))

    events = list(
        stream_run(
            ChatRequest(message="Credit C-1001 $75 for the outage"),
            settings=sandbox,
            graph=graph,
        )
    )
    kinds = [e["event"] for e in events]
    assert "error" not in kinds

    done = [e for e in events if e["event"] == "done"][0]
    response = done["response"]
    assert response["status"] == "awaiting_approval"
    pending = [a for a in response["approvals"] if a["status"] == "pending"]
    assert [a["action"] for a in pending] == ["apply_account_credit"]
    assert response["answer"], "the hold message should be delivered as the answer"


def test_graph_error_surfaces_as_a_single_error_event(sandbox):
    """A crash must not look like a successful empty answer."""
    graph = _build(sandbox)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic responder failure")

    llm = ScriptedLLM(_boom, lambda system, human, node: ANSWER, name="boom")
    # Rebuild so the failing responder is the one the graph uses.
    from app.graph.builder import build_graph

    graph = build_graph(sandbox, llm=llm, memory=get_memory_store(sandbox))

    events = list(
        stream_run(ChatRequest(message=QUESTION), settings=sandbox, graph=graph)
    )
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "error"
    assert "done" not in kinds
    assert "synthetic responder failure" in [e for e in events if e["event"] == "error"][0][
        "error"
    ]


def test_non_streaming_respond_node_is_unchanged(sandbox):
    """The blocking path must not need a stream writer at all."""
    graph = _build(sandbox)
    response = run(ChatRequest(message=QUESTION), settings=sandbox, graph=graph)
    assert response.answer.strip() == ANSWER
    assert response.status == "completed"


@pytest.mark.parametrize("llm_mode", ["groq-adapter"])
def test_groq_adapter_stream_is_a_generator(sandbox, llm_mode):
    """The real adapter must expose stream_text without a key or network."""
    from app.llm import GroqStructuredLLM

    llm = object.__new__(GroqStructuredLLM)
    llm.settings = sandbox
    llm.model_name = "openai/gpt-oss-20b"

    class _Chunk:
        def __init__(self, content, tin=0, tout=0):
            self.content = content
            self.usage_metadata = {"input_tokens": tin, "output_tokens": tout}

    class _StreamingChat:
        def stream(self, messages):
            yield _Chunk("Hello")
            yield _Chunk("", tin=12, tout=7)  # usage rides on a later chunk
            yield _Chunk(" world")

    llm._chat = _StreamingChat()
    usage: dict[str, int] = {}
    parts = list(llm.stream_text(system="s", human="h", node="respond", usage=usage))

    assert "".join(parts) == "Hello world"
    assert usage == {"in": 12, "out": 7}
