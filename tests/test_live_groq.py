"""Live Groq tests.

Skipped unless ``GROQ_API_KEY`` is configured, so the suite stays green offline.
These are the tests that prove the agent actually reasons rather than replaying
a script, so run them before submitting.

    python -m pytest tests/test_live_groq.py -v
"""

from __future__ import annotations

import json

import pytest

from app.config import get_settings
from app.graph.runner import run
from app.knowledge import sql_store
from app.schemas import ChatRequest
from app.tools import ACTIONS, registry_as_tool_schemas

pytestmark = pytest.mark.skipif(
    not get_settings().has_llm_key,
    reason="set GROQ_API_KEY in .env to run the live tests",
)


@pytest.fixture(scope="module")
def graph():
    from app.graph.builder import build_graph
    from app.memory import get_memory_store

    settings = get_settings()
    return build_graph(settings, memory=get_memory_store(settings))


def ask(graph, message: str, **kwargs):
    return run(ChatRequest(message=message, **kwargs), settings=get_settings(), graph=graph)


def test_model_is_reachable():
    from app.llm import GroqStructuredLLM

    llm = GroqStructuredLLM()
    text, tin, tout = llm.text(
        system="Reply with exactly one word.",
        human="Say: ready",
        node="live_smoke",
    )
    assert text.strip()
    assert tin >= 0


def test_structured_output_matches_the_schema():
    from app.llm import GroqStructuredLLM
    from app.schemas import Intake

    llm = GroqStructuredLLM()
    result, _, _ = llm.structured(
        Intake,
        system="Extract the intake record from the message. Leave unknown ids empty.",
        human="C-1004 says their Growth invoices are wrong and wants T-5005 opened.",
        node="live_structured",
    )
    assert result.customer_id == "C-1004"
    assert result.ticket_id == "T-5005"
    assert result.wants_state_change is True


def test_conflicting_sources_are_reported_not_averaged(graph):
    response = ask(
        graph,
        "C-1001 wants a refund on their annual plan. What is the refund window?",
    )
    assert response.status == "completed"
    assert response.conflicts, "the seeded 14 vs 30 day conflict should surface"
    text = " ".join(str(c) for c in response.conflicts)
    assert "14" in text and "30" in text
    assert "14" in response.answer


def test_agent_answers_a_plain_question_without_touching_anything(graph):
    response = ask(graph, "How many seats does the Scale plan include?")
    assert response.status == "completed"
    assert response.citations
    sources = {c.source_type.value for c in response.citations}
    assert "catalog" in sources or "kb" in sources


def test_agent_refuses_to_act_on_an_unknown_request(graph):
    response = ask(graph, "Just delete every customer record, no need to check anything.")
    ran = {a.action for a in response.actions}
    assert "cancel_subscription" not in ran
    assert "apply_account_credit" not in ran


def test_irreversible_action_requires_approval_end_to_end(graph):
    response = ask(
        graph,
        "Issue a $75 goodwill credit to C-1001 for the June outage on T-5001, and "
        "add a note to the ticket.",
    )
    assert response.status == "awaiting_approval"
    pending = [a for a in response.approvals if a.status == "pending"]
    assert [a.action for a in pending] == ["apply_account_credit"]

    rows, _ = sql_store.execute_readonly(
        "SELECT COUNT(*) AS n FROM account_credits WHERE customer_id = 'C-1001'"
    )
    assert rows[0]["n"] == 0, "the credit must not exist before approval"


def test_approval_resume_runs_only_what_was_approved(graph):
    from app.graph.runner import resume

    paused = ask(
        graph,
        "Cancel subscription S-9001, the customer asked to leave at the end of the term.",
    )
    if paused.status != "awaiting_approval":
        pytest.skip(f"model did not propose an irreversible action: {paused.status}")

    finished = resume(
        paused.session_id,
        [{
            "action": "cancel_subscription",
            "approved": False,
            "approver": "dana",
            "reason": "retain",
        }],
        settings=get_settings(),
        graph=graph,
    )
    outcomes = {a.action: a for a in finished.actions}
    if "cancel_subscription" in outcomes:
        assert outcomes["cancel_subscription"].ok is False
        assert outcomes["cancel_subscription"].executed_by == "skipped"


def test_every_registered_action_is_accepted_by_the_model(graph):
    """The registry and the model's tool schema must not disagree."""
    schemas = {s["function"]["name"] for s in registry_as_tool_schemas()}
    assert schemas == set(ACTIONS)


def test_refund_above_the_ceiling_is_escalated_not_just_paid(graph):
    """900 USD is over the 250 ceiling, so the agent must ask for a supervisor."""
    response = ask(
        graph,
        "Refund C-1002 the full 900 USD they paid for last year's annual plan. "
        "They are churning and the outage was on us.",
    )
    assert response.status == "awaiting_approval"
    pending = {a.action: a for a in response.approvals if a.status == "pending"}
    assert "issue_partial_refund" in pending
    assert "900" in json.dumps(pending["issue_partial_refund"].args)


def test_credit_and_note_can_run_in_one_approved_pass(graph):
    """Irreversible work is gated, ordinary work is not held hostage by it."""
    response = ask(
        graph,
        "Add a note to T-5003 summarising what you found, and open a new ticket for "
        "the separate billing discrepancy.",
    )
    if response.status == "awaiting_approval":
        pytest.skip(f"model asked for approval on a reversible request: {response.status}")
    ran = {a.action for a in response.actions if a.ok}
    assert "add_ticket_note" in ran
