"""The two new HTTP surfaces: SSE streaming and user-managed KB documents.

Both are contract tests. The agent is scripted here, so the stream is real
without a provider key or network, and the KB is the sandboxed data directory.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

# Reuse the API client fixtures (with and without a key) rather than
# duplicating the module-reload dance they already encapsulate. They are
# aliased because pytest injects them as parameters named `client`, which
# would otherwise read as a redefinition of this import.
from tests.test_api import bare_client as _bare_client
from tests.test_api import client as _client

client = _client
bare_client = _bare_client


def _events(api: TestClient, path: str, payload: dict) -> list[dict]:
    """Collect the SSE frames from a streamed response."""
    with api.stream("POST", path, json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # A proxy that buffers this defeats streaming, so the header is part of
        # the contract rather than a nicety.
        assert response.headers.get("x-accel-buffering") == "no"
        frames = []
        for line in response.iter_lines():
            if line.startswith("data:"):
                frames.append(json.loads(line[5:].strip()))
        return frames


@pytest.fixture
def scripted_client(sandbox, monkeypatch):
    """A client whose graph runs a scripted LLM, so streaming is deterministic."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AGENT_LLM_MODE", "fake")
    from app.config import get_settings

    get_settings.cache_clear()
    import app.api.main as main
    import app.graph.builder as builder
    import app.llm as llm_mod
    from app.memory import get_memory_store

    get_settings.cache_clear()

    from tests.test_graph import make_responder

    script = llm_mod.ScriptedLLM(
        make_responder(),
        lambda system, human, node: "Streamed answer with a citation.",
        name="test",
    )
    original = builder.build_graph
    builder.build_graph = lambda s, **kw: original(s, llm=script, memory=get_memory_store(s))
    main.build_graph = lambda s, **kw: original(s, llm=script, memory=get_memory_store(s))

    importlib.reload(main)
    main.build_graph = lambda s, **kw: original(s, llm=script, memory=get_memory_store(s))

    with TestClient(main.app) as test_client:
        yield test_client

    builder.build_graph = original
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #


def test_stream_returns_sse_events(scripted_client):
    events = _events(
        scripted_client,
        "/chat/stream",
        {"message": "What is the refund window for an annual plan?"},
    )
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "token" in kinds
    assert "error" not in kinds


def test_stream_tokens_reassemble_into_the_done_answer(scripted_client):
    events = _events(
        scripted_client,
        "/chat/stream",
        {"message": "What is the refund window for an annual plan?"},
    )
    streamed = "".join(e["text"] for e in events if e["event"] == "token")
    done = [e for e in events if e["event"] == "done"][0]["response"]
    # The scripted responder splits on spaces and rejoins with a trailing space
    # per token, so the deltas carry whitespace the final answer is stripped of.
    assert streamed.strip() == done["answer"].strip()
    assert done["answer"] == "Streamed answer with a citation."


def test_stream_reports_node_progress(scripted_client):
    events = _events(
        scripted_client,
        "/chat/stream",
        {"message": "What is the refund window for an annual plan?"},
    )
    nodes = [e["node"] for e in events if e["event"] == "node"]
    assert "intake" in nodes
    assert "respond" in nodes


def test_stream_rejects_an_empty_message(scripted_client):
    response = scripted_client.post("/chat/stream", json={"message": "   "})
    assert response.status_code == 422


def test_stream_refuses_without_a_key(bare_client):
    response = bare_client.post("/chat/stream", json={"message": "hello"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_stream_and_blocking_chat_agree(scripted_client):
    """The fallback endpoint must not drift from the streaming one."""
    message = "What is the refund window for an annual plan?"
    blocked = scripted_client.post("/chat", json={"message": message}).json()
    events = _events(scripted_client, "/chat/stream", {"message": message})
    done = [e for e in events if e["event"] == "done"][0]["response"]

    assert done["answer"] == blocked["answer"]
    assert done["status"] == blocked["status"]
    assert done["session_id"] != blocked["session_id"], "each turn gets its own session"

    # Compare the trace's shape, not its literal text: timestamps and per-call
    # latencies legitimately differ between two independent runs.
    def shape(trace: list[dict]) -> list[tuple]:
        return [(s["node"], s["label"]) for s in trace]

    assert shape(done["trace"]) == shape(blocked["trace"])


# --------------------------------------------------------------------------- #
# source management
# --------------------------------------------------------------------------- #

DOC = """# Escalation matrix

Tier 1 owns triage during business hours. Tier 2 takes Sev-1 incidents.
"""


def test_list_documents_seeds_the_kb(client):
    body = client.get("/sources/documents").json()
    assert body["count"] >= 1
    names = {d["name"] for d in body["documents"]}
    assert "billing-and-refunds.md" in names
    for doc in body["documents"]:
        assert doc["chunks"] >= 1, f"{doc['name']} reported no indexed chunks"


def test_add_document_writes_indexes_and_lists(client):
    created = client.post(
        "/sources/documents", json={"name": "escalation.md", "content": DOC}
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "escalation.md"
    assert body["created"] is True
    assert body["indexed_chunks"] >= 2

    names = {d["name"] for d in client.get("/sources/documents").json()["documents"]}
    assert "escalation.md" in names


def test_added_document_becomes_retrievable(client, sandbox):
    """The point of indexing: the new text must be chunked into the collection.

    Deliberately asserted through the collection rather than a similarity
    ranking. The offline suite forces the hashing embedding backend, whose
    near-uniform scores make rank order an artefact of the backend, and HNSW
    does not guarantee that a freshly added vector is reachable from every
    entry point. What must hold is that the document is indexed and its text
    is present; that it *ranks* well is a property of the real backend and
    belongs in a live test.
    """
    from app.knowledge.vector_store import get_vector_store

    client.post("/sources/documents", json={"name": "escalation.md", "content": DOC})

    store = get_vector_store(sandbox)
    stored = store.get(where={"source": "escalation.md"}, include=["documents"])
    ids = stored.get("ids") or []
    assert ids, "the new document produced no chunks"
    body = " ".join(stored["documents"])
    assert "Tier 2 takes Sev-1 incidents" in body

    # And the API agrees it is part of the KB.
    docs = client.get("/sources/documents").json()["documents"]
    row = next(d for d in docs if d["name"] == "escalation.md")
    assert row["chunks"] == len(ids)


def test_add_document_refuses_a_duplicate_without_overwrite(client):
    client.post("/sources/documents", json={"name": "escalation.md", "content": DOC})
    clash = client.post(
        "/sources/documents", json={"name": "escalation.md", "content": "different"}
    )
    assert clash.status_code == 400
    assert "already exists" in clash.json()["detail"]


def test_overwrite_replaces_the_document(client, sandbox):
    client.post("/sources/documents", json={"name": "escalation.md", "content": DOC})
    updated = client.post(
        "/sources/documents",
        json={
            "name": "escalation.md",
            "content": "# Only heading\n\nSolo tier.",
            "overwrite": True,
        },
    )
    assert updated.status_code == 201
    assert updated.json()["created"] is False

    from app.knowledge.vector_store import get_vector_store

    store = get_vector_store(sandbox)
    stored = store.get(where={"source": "escalation.md"}, include=["documents"])
    body = " ".join(stored.get("documents") or [])
    # Chunk ids are derived from content, so re-indexing alone would leave the
    # old chunks in the collection still answering retrieval.
    assert "Tier 2" not in body, "the superseded text is still indexed"
    assert "Solo tier" in body


def test_delete_document_removes_file_and_purges_vectors(client, sandbox):
    from app.knowledge.vector_store import chunk_counts_by_source

    client.post("/sources/documents", json={"name": "escalation.md", "content": DOC})
    assert chunk_counts_by_source(sandbox).get("escalation.md", 0) >= 1

    deleted = client.delete("/sources/documents/escalation.md")
    assert deleted.status_code == 200
    assert deleted.json()["removed_chunks"] >= 1

    names = {d["name"] for d in client.get("/sources/documents").json()["documents"]}
    assert "escalation.md" not in names
    # This is the invariant that matters: index_kb only upserts on stable chunk
    # ids, so deleting the markdown file alone would leave its vectors in the
    # collection, still retrievable and now owned by no file.
    assert "escalation.md" not in chunk_counts_by_source(sandbox)


def test_delete_unknown_document_is_404(client):
    assert client.delete("/sources/documents/nope.md").status_code == 404


@pytest.mark.parametrize(
    "name",
    [
        "../escape.md",
        "..\\escape.md",
        "sub/dir.md",
        "/etc/passwd",
        ".hidden.md",
        "",
        "   ",
        "nul.md",
        "a" * 200 + ".md",
    ],
)
def test_unsafe_document_names_are_rejected(client, name):
    assert (
        client.post("/sources/documents", json={"name": name, "content": DOC}).status_code
        == 400
    )
    # An empty or whitespace name collapses the path to the collection root,
    # which is a routing miss (405) rather than a validation failure.
    delete_status = client.delete(f"/sources/documents/{name}").status_code
    assert delete_status in {400, 404, 405, 422}


def test_empty_document_is_rejected(client):
    response = client.post("/sources/documents", json={"name": "empty.md", "content": "   "})
    assert response.status_code == 400


def test_name_without_extension_gets_md(client):
    body = client.post(
        "/sources/documents", json={"name": "no-extension", "content": DOC}
    ).json()
    assert body["name"] == "no-extension.md"


def test_source_endpoints_need_no_llm_key(bare_client):
    """Managing the KB must work even when reasoning cannot."""
    assert bare_client.get("/sources/documents").status_code == 200
    assert (
        bare_client.post(
            "/sources/documents", json={"name": "no-key.md", "content": DOC}
        ).status_code
        == 201
    )
