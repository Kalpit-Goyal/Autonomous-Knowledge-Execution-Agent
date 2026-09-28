"""HTTP surface.

The reasoning is stubbed here; what is under test is the contract the Streamlit
client depends on - status codes, the approval round trip, and the shape of the
diagnostics payloads.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(sandbox, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    from app.config import get_settings

    # The sandbox fixture already populated the settings cache, so clear it
    # before the module under test re-reads the environment.
    get_settings.cache_clear()
    import app.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as test_client:
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def bare_client(sandbox, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    import app.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["llm_model"]


def test_chat_refuses_without_a_key(bare_client):
    response = bare_client.post("/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]


def test_chat_rejects_an_empty_message(client):
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_sources_lists_all_four(client):
    body = client.get("/sources").json()
    assert set(body) == {"knowledge_base", "operations_db", "policy", "catalog"}
    assert body["knowledge_base"]["indexed"]["documents"] > 0
    assert "customers" in body["operations_db"]["schema"]


def test_sources_shape_matches_what_the_sidebar_reads(client):
    """Pin the exact fields `app/ui/app.py` indexes, and their types.

    The sidebar crashed on launch once because `catalog.rows` was treated as a
    list when it is a count. A key-existence check is not enough, so the types
    are asserted too.
    """
    body = client.get("/sources").json()

    indexed = body["knowledge_base"]["indexed"]
    assert isinstance(indexed["documents"], int)
    assert isinstance(indexed["backend"], str)

    assert isinstance(body["operations_db"]["schema"], str)

    assert isinstance(body["policy"]["version"], str)
    assert isinstance(body["policy"]["sections"], list)
    assert isinstance(body["policy"]["approver_roles"], list)
    assert body["policy"]["approver_roles"], "the UI offers these as a dropdown"

    catalog = body["catalog"]
    assert isinstance(catalog["rows"], int), "rows is a count, not a list of rows"
    assert isinstance(catalog["plans"], list)
    assert all(isinstance(p, str) for p in catalog["plans"])


def test_stats_shape_matches_what_the_sidebar_reads(client):
    body = client.get("/stats").json()
    assert isinstance(body["audit"]["total"], int)
    assert isinstance(body["approvals"]["pending"], int)
    assert isinstance(body["approvals"]["waiting_on"], list)
    assert isinstance(body["knowledge"]["documents"], int)
    assert isinstance(body["catalog"]["rows"], int)
    assert isinstance(body["rows"], dict)
    assert all(isinstance(v, int) for v in body["rows"].values())


def test_pending_approvals_shape_matches_what_the_sidebar_reads(client):
    body = client.get("/approvals/pending").json()
    assert isinstance(body, list)
    for item in body:
        assert isinstance(item["action"], str)
        assert "session_id" in item


def test_actions_endpoint_exposes_the_catalogue(client):
    body = client.get("/actions").json()
    assert len(body["actions"]) == 10
    irreversible = {a["name"] for a in body["actions"] if a["irreversible"]}
    assert irreversible == {
        "apply_account_credit",
        "cancel_subscription",
        "issue_partial_refund",
    }
    for action in body["actions"]:
        assert action["description"]
        assert action["effect"]
        assert action["parameters"]


def test_stats_reports_real_row_counts(client):
    body = client.get("/stats").json()
    assert body["rows"]["customers"] == 10
    assert body["rows"]["tickets"] == 12
    assert body["knowledge"]["documents"] > 0
    assert body["approvals"]["pending"] == 0


def test_audit_endpoint_filters_by_event_type(client):
    body = client.get("/audit?event_type=intake&limit=5").json()
    assert isinstance(body, list)


def test_approving_without_a_pending_request_is_404(client):
    assert client.post("/chat/NOPE/approve", json={"approved": True}).status_code == 404


def test_pending_approvals_endpoint(client):
    body = client.get("/approvals/pending").json()
    assert isinstance(body, list)


def test_unknown_path_is_404(client):
    assert client.get("/does-not-exist").status_code == 404
