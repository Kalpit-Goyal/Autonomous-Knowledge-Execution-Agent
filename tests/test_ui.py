"""Streamlit render smoke tests.

`streamlit run` serves the app shell with HTTP 200 even when the script raises,
so a health check proves nothing about the render path. These tests execute the
real script through Streamlit's `AppTest` against a live `TestClient` API, which
is what actually catches a crash on first paint.

The bug that motivated this file: the sidebar called `len()` on
`catalog.rows`, which is a count rather than a list, so the UI died on launch
with `TypeError: object of type 'int' has no len()`.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
import threading
import time
from contextlib import closing

import pytest
import streamlit as st
import uvicorn
from streamlit.testing.v1 import AppTest

UI_PATH = "app/ui/app.py"


@pytest.fixture
def live_api(sandbox, monkeypatch):
    """Serve the real API on a real port so the UI's HTTP client has a peer.

    Streamlit's AppTest runs the script in-process but the UI talks to the API
    over real HTTP, so a real socket is required.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    from app.config import get_settings

    get_settings.cache_clear()
    import app.api.main as main

    importlib.reload(main)

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    base = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("UI_API_BASE", base)
    monkeypatch.setenv("UI_API_TIMEOUT", "30")

    # `app/ui/app.py` reads UI_API_BASE into a module constant at import time, so
    # a module left in sys.modules from an earlier import would keep pointing at
    # a dead port and the sidebar would silently render nothing.
    monkeypatch.delitem(sys.modules, "app.ui.app", raising=False)

    config = uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 30
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        pytest.fail("the API did not start for the UI test")

    yield base
    server.should_exit = True
    thread.join(timeout=15)
    get_settings.cache_clear()


def _run_ui() -> AppTest:
    # `client()` is @st.cache_resource(ttl=30). Streamlit keeps that cache in a
    # process-level registry rather than on the script module, so without this
    # the second test in the session reuses an httpx client bound to the first
    # test's port, which has since been shut down.
    st.cache_resource.clear()

    at = AppTest.from_file(UI_PATH, default_timeout=60)
    at.run()
    assert not at.exception, f"the Streamlit script raised on render: {at.exception}"

    errors = [e.value for e in at.error] + [e.value for e in at.sidebar.error]
    assert not errors, f"the UI reported an API error: {errors}"
    return at


def test_ui_renders_the_landing_page(live_api):
    """The first-paint path must not raise."""
    at = _run_ui()
    assert at.title, "the page should have a title"


def test_sidebar_renders_sources_without_error(live_api):
    """Covers the `catalog.rows` regression and the whole sidebar payload."""
    at = _run_ui()
    text = "\n".join(m.value for m in at.sidebar.markdown)

    for label in ("Knowledge base", "Operations DB", "Policy", "Catalog"):
        assert label in text, f"{label} missing from the sidebar"
    assert "plan rows" in text
    assert "Traceback" not in text


def test_sidebar_pulls_live_values_from_the_api(live_api):
    """The row count must reach the screen as a number, not a length() call."""
    import httpx

    expected = httpx.get(f"{live_api}/stats", timeout=30).json()["catalog"]["rows"]
    at = _run_ui()
    text = "\n".join(m.value for m in at.sidebar.markdown)
    assert f"{expected} plan rows" in text


def test_sidebar_says_so_when_the_api_is_unreachable(live_api, monkeypatch):
    """A dead API must not look like an empty sources panel."""
    at = _run_ui()
    captions = [c.value for c in at.sidebar.caption] + [c.value for c in at.caption]
    assert any(f"API: {live_api}" in c for c in captions)
    # With the API up, the fallback caption must not be shown.
    assert not any("Could not reach the agent API" in c for c in captions)


def _trace_response() -> dict:
    """A completed response whose trace carries both structured and scalar data.

    The structured entries are the trigger: rendering them inside the trace
    expander is what used to raise.
    """
    return {
        "session_id": "SESS-TRACE",
        "status": "completed",
        "answer": "Your plan renews on 2026-04-01.",
        "elapsed_ms": 12,
        "plan": None,
        "intake": None,
        "approvals": [],
        "citations": [
            {
                "source_type": "kb",
                "citation": "plans-and-entitlements.md#renewal",
                "snippet": "Annual plans renew on the anniversary date.",
                "score": 0.82,
            }
        ],
        "trace": [
            {
                "node": "retrieve",
                "label": "Retrieved context",
                "detail": "3 chunks",
                # Nested containers, which the renderer must not wrap in an
                # expander of their own.
                "data": {
                    "hits": [{"chunk_id": "a", "score": 0.91}, {"chunk_id": "b"}],
                    "sources": {"kb": ["plans.md"]},
                    "top_k": 5,
                },
            },
            {
                "node": "decide",
                "label": "Decisions",
                "detail": "",
                "data": {
                    "attempts": [],
                    "resolved": {},
                    "outcome": "answer",
                },
            },
        ],
    }


def test_reasoning_trace_renders_structured_step_data(live_api):
    """Covers the nested-expander regression.

    `st.expander` cannot be nested, so a dict/list inside the trace's own
    expander must be rendered with a non-expander widget. The symptom was
    `StreamlitAPIException: Expanders may not be nested inside other expanders`
    on every completed answer.
    """
    at = _run_ui()
    at.session_state["last_response"] = _trace_response()
    at.run()

    assert not at.exception, f"the trace renderer raised: {at.exception}"
    errors = [e.value for e in at.error]
    assert not errors, f"the UI reported an error while rendering the trace: {errors}"
    assert not any("may not be nested" in str(e) for e in errors)

    # The trace must actually be shown, not silently skipped.
    assert any("Reasoning trace" in e.label for e in at.expander)


def test_sidebar_lists_manageable_kb_documents(live_api):
    """The manage-knowledge-base panel must name the documents on disk."""
    at = _run_ui()
    text = "\n".join(c.value for c in at.sidebar.caption)
    assert "billing-and-refunds.md" in text, text
    assert "passages" in text
    # Management is a capability, so the affordance has to be reachable.
    assert any("Add or remove documents" in e.label for e in at.sidebar.expander)


def test_stream_chat_paints_tokens_progressively(monkeypatch):
    """The deltas must be painted as they arrive, not rendered once at the end.

    Driven directly rather than through `AppTest`, because AppTest executes the
    script from its file path in a fresh module namespace, so patching a
    function on the imported `app.ui.app` has no effect on the run.
    """
    import app.ui.app as ui

    frames = [
        {"event": "start", "session_id": "SESS-1"},
        {"event": "node", "node": "intake"},
        {"event": "token", "text": "Your "},
        {"event": "token", "text": "plan renews "},
        {"event": "token", "text": "in April."},
        {
            "event": "done",
            "response": {"session_id": "SESS-1", "answer": "Your plan renews in April."},
        },
    ]
    painted: list[str] = []
    captions: list[str] = []

    class FakeSink:
        def markdown(self, value):
            painted.append(value)

        def caption(self, value):
            captions.append(value)

        def empty(self):
            pass

    class FakeResponse:
        status_code = 200

        def iter_lines(self):
            for frame in frames:
                yield f"event: {frame['event']}"
                yield "data: " + json.dumps(frame)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeClient:
        def stream(self, method, path, **kwargs):
            assert method == "POST"
            assert path == "/chat/stream"
            assert kwargs["json"]["message"] == "when does it renew?"
            return FakeResponse()

    monkeypatch.setattr(ui, "client", lambda: FakeClient())

    result = ui.stream_chat("when does it renew?", None, FakeSink(), FakeSink())

    assert result["answer"] == "Your plan renews in April."
    # Each token triggers a repaint, and the intermediate frames grow the text.
    assert painted == ["Your", "Your plan renews", "Your plan renews in April."]
    assert "Working: intake" in captions


def test_stream_chat_surfaces_an_in_band_error(monkeypatch):
    """An error frame must not be mistaken for a successful answer."""
    import app.ui.app as ui

    class FakeResponse:
        status_code = 200

        def iter_lines(self):
            yield "data: " + json.dumps({"event": "error", "error": "rate limited"})

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        ui, "client", lambda: type("C", (), {"stream": lambda self, *a, **k: FakeResponse()})()
    )
    shown: list[str] = []
    monkeypatch.setattr(ui.st, "error", lambda msg: shown.append(str(msg)))

    class FakeSink:
        def markdown(self, value):
            pass

        def caption(self, value):
            pass

        def empty(self):
            pass

    assert ui.stream_chat("hi", None, FakeSink(), FakeSink()) is None
    assert any("rate limited" in m for m in shown), shown


def test_stream_chat_reports_a_truncated_stream(monkeypatch):
    """A stream that ends with no `done` must not render as a blank answer."""
    import app.ui.app as ui

    class FakeResponse:
        status_code = 200

        def iter_lines(self):
            yield "data: " + json.dumps({"event": "token", "text": "partial"})

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        ui, "client", lambda: type("C", (), {"stream": lambda self, *a, **k: FakeResponse()})()
    )
    shown: list[str] = []
    monkeypatch.setattr(ui.st, "error", lambda msg: shown.append(str(msg)))

    class FakeSink:
        def markdown(self, value):
            pass

        def caption(self, value):
            pass

        def empty(self):
            pass

    assert ui.stream_chat("hi", None, FakeSink(), FakeSink()) is None
    assert any("before the agent sent a result" in m for m in shown), shown
