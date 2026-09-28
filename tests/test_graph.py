"""The graph, driven by a scripted LLM.

These tests exercise the real graph: real retrieval, real SQL, real writes, real
interrupts. Only the reasoning is stubbed, which is what makes the control flow
testable at all.
"""

from __future__ import annotations

from app.audit import query as audit_query
from app.graph.runner import resume, run
from app.knowledge import sql_store
from app.memory import get_memory_store
from app.schemas import (
    ChatRequest,
    Decision,
    DecisionSet,
    Intake,
    Plan,
    PlanStep,
    ReconcileStatus,
    Reconciliation,
    RetrievalTask,
    SourceType,
    Verification,
)

# --------------------------------------------------------------------------- #
# scripted reasoning
# --------------------------------------------------------------------------- #


def make_responder(**overrides):
    """Build a responder whose per-schema outputs can be overridden per test."""
    defaults = {
        "intake": Intake(
            intent="complaint",
            summary="C-1001 churn risk over an outage and a refund dispute",
            customer_id="C-1001",
            ticket_id="T-5001",
            priority="P2",
            wants_state_change=True,
        ),
        "plan": Plan(
            goal="Assess the churn risk and handle the refund dispute correctly.",
            steps=[
                PlanStep(
                    id="state",
                    goal="Read the account and ticket state",
                    why="Refund eligibility depends on tenure.",
                    sources=[SourceType.OPS_DB],
                    tasks=[
                        RetrievalTask(
                            step_id="state",
                            source_type=SourceType.OPS_DB,
                            query="C-1001 account",
                            purpose="live state",
                            sql=(
                                "SELECT customer_id, plan FROM customers "
                                "WHERE customer_id = 'C-1001'"
                            ),
                        )
                    ],
                ),
                PlanStep(
                    id="policy",
                    goal="Find the governing refund rule",
                    why="The knowledge base and the policy disagree.",
                    sources=[SourceType.POLICY, SourceType.KB],
                    tasks=[
                        RetrievalTask(
                            step_id="policy",
                            source_type=SourceType.POLICY,
                            query="annual refund window",
                            purpose="authority",
                        ),
                        RetrievalTask(
                            step_id="policy_kb",
                            source_type=SourceType.KB,
                            query="refund window days",
                            purpose="documentation",
                        ),
                    ],
                ),
            ],
        ),
        "reconcile": Reconciliation(
            status=ReconcileStatus.PROCEED,
            reason="Policy says 14 days for annual, the knowledge base says 30. Policy governs.",
            conflicts=["Refund window: policy 14 days vs knowledge base 30 days"],
        ),
        "decide": DecisionSet(
            decisions=[],
            answer_without_action=(
                "Annual plans are non-refundable after 14 days per the policy of record, "
                "even though the knowledge base still says 30."
            ),
        ),
        "verify": Verification(
            achieved=True, confidence=0.9, reasoning="The note was recorded."
        ),
    }
    defaults.update(overrides)

    def responder(schema, system, human, node):
        if schema is Intake:
            return defaults["intake"]
        if schema is Plan:
            return defaults["plan"]
        if schema is Reconciliation:
            return defaults["reconcile"]
        if schema is DecisionSet:
            return defaults["decide"]
        if schema is Verification:
            return defaults["verify"]
        raise AssertionError(f"unscripted schema {schema.__name__} at node {node}")

    return responder


def build(sandbox, **overrides):
    from app.graph.builder import build_graph
    from app.llm import ScriptedLLM

    llm = ScriptedLLM(
        make_responder(**overrides),
        lambda system, human, node: "Scripted answer with a citation.",
        name="test",
    )
    return build_graph(sandbox, llm=llm, memory=get_memory_store(sandbox))


def _invoke(graph, message, session_id, **kwargs):
    """Run one turn through the graph, with a unique thread id per call."""
    import uuid

    from app.config import get_settings
    from app.schemas import ChatRequest

    return run(
        ChatRequest(
            message=message, session_id=f"{session_id}-{uuid.uuid4().hex[:6]}", **kwargs
        ),
        settings=get_settings(),
        graph=graph,
    )


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #


def test_answer_only_request_takes_no_action(sandbox):
    graph = build(sandbox)
    response = _invoke(graph, "What is the refund window for an annual plan?", "S-Q")
    assert response.status == "completed"
    assert response.answer
    assert response.actions == []
    assert response.decisions == []
    assert response.citations
    assert response.trace


def test_every_node_appears_in_the_trace(sandbox):
    graph = build(sandbox)
    response = _invoke(graph, "Assess C-1001", "S-TRACE")
    nodes = [t.node for t in response.trace]
    expected_nodes = (
        "intake", "plan", "retrieve", "reconcile", "decide", "execute", "verify", "respond"
    )
    for expected in expected_nodes:
        assert expected in nodes, f"{expected} missing from {nodes}"


def test_retrieval_covered_every_planned_source(sandbox):
    graph = build(sandbox)
    response = _invoke(graph, "Assess C-1001", "S-SRC")
    assert {c.source_type.value for c in response.citations} >= {"ops_db", "policy", "kb"}


def test_conflict_is_recorded_and_surfaced(sandbox):
    graph = build(sandbox)
    response = _invoke(graph, "Assess C-1001", "S-CONF")
    assert response.conflicts
    assert "14" in response.conflicts[0]["summary"]


def test_run_is_fully_audited(sandbox):
    graph = build(sandbox, decide=_note_decision_set())
    response = _invoke(graph, "Assess C-1001 and log it", "S-AUDIT")
    events = {
        row["event_type"]
        for row in audit_query(session_id=response.session_id, settings=sandbox)
    }
    assert {
        "intake", "plan", "retrieve", "reconcile",
        "decide", "action_succeeded", "verify", "respond",
    } <= events


# --------------------------------------------------------------------------- #
# approval
# --------------------------------------------------------------------------- #

CREDIT = Decision(
    id="credit",
    action="apply_account_credit",
    args={"customer_id": "C-1001", "amount_usd": 75.0, "reason": "outage", "ticket_id": "T-5001"},
    why="Goodwill for the outage.",
    risk="high",
)

# Above the 250 USD ceiling in policies.json, so the approver must also present
# an allowed role.
BIG_REFUND = Decision(
    id="refund",
    action="issue_partial_refund",
    args={
        "customer_id": "C-1001",
        "amount_usd": 400.0,
        "reason": "annual plan cancelled in month 2",
        "ticket_id": "T-5001",
    },
    why="Customer is owed a pro-rated refund.",
    risk="high",
)


def _refund_decision_set():
    return DecisionSet(decisions=[BIG_REFUND.model_copy(deep=True)])


def _note_decision_set() -> DecisionSet:
    return DecisionSet(
        decisions=[
            Decision(
                id="note",
                action="add_ticket_note",
                args={"ticket_id": "T-5001", "body": "investigated"},
                why="Record the investigation.",
            )
        ]
    )


def _credit_decision_set():
    return DecisionSet(
        decisions=[
            Decision(
                id="note",
                action="add_ticket_note",
                args={"ticket_id": "T-5001", "body": "investigated"},
                why="Record the investigation.",
            ),
            CREDIT.model_copy(deep=True),
        ]
    )


def test_irreversible_action_pauses_for_approval(sandbox):
    graph = build(sandbox, decide=_credit_decision_set())
    response = _invoke(graph, "Credit C-1001 $75 for the outage", "S-APPR")

    assert response.status == "awaiting_approval"
    pending = [a for a in response.approvals if a.status == "pending"]
    assert [a.action for a in pending] == ["apply_account_credit"]
    assert "irreversible" in pending[0].reason_for_approval.lower()

    ran = {a.action for a in response.actions}
    assert "apply_account_credit" not in ran, "the credit must not run before approval"
    assert "add_ticket_note" not in ran, "nothing should run while paused"

    rows, _ = sql_store.execute_readonly(
        "SELECT COUNT(*) AS n FROM account_credits", settings=sandbox
    )
    assert rows[0]["n"] == 0


def test_approval_carries_the_approver_into_the_credit(sandbox):
    graph = build(sandbox, decide=_credit_decision_set())
    response = _invoke(graph, "Credit C-1001 $75 for the outage", "S-RES")
    assert response.status == "awaiting_approval"

    finished = resume(
        response.session_id,
        [{
            "action": "apply_account_credit",
            "approved": True,
            "approver": "dana",
            "reason": "fair",
        }],
        settings=sandbox,
        graph=graph,
    )
    assert finished.status == "completed"

    outcomes = {a.action: a for a in finished.actions}
    assert outcomes["apply_account_credit"].ok
    assert outcomes["apply_account_credit"].result["approved_by"] == "dana"
    assert outcomes["add_ticket_note"].ok, "the reversible action should still have run"

    rows, _ = sql_store.execute_readonly(
        "SELECT amount_usd, approved_by FROM account_credits", settings=sandbox
    )
    assert len(rows) == 1
    assert abs(rows[0]["amount_usd"] - 75.0) < 0.01
    assert rows[0]["approved_by"] == "dana"


def test_rejection_prevents_the_write_and_is_explained(sandbox):
    graph = build(sandbox, decide=_credit_decision_set())
    response = _invoke(graph, "Credit C-1001 $75 for the outage", "S-REJ")

    finished = resume(
        response.session_id,
        [{"action": "apply_account_credit", "approved": False, "approver": "dana", "reason": "no"}],
        settings=sandbox,
        graph=graph,
    )
    outcomes = {a.action: a for a in finished.actions}
    assert outcomes["apply_account_credit"].ok is False
    assert outcomes["apply_account_credit"].executed_by == "skipped"

    rows, _ = sql_store.execute_readonly(
        "SELECT COUNT(*) AS n FROM account_credits", settings=sandbox
    )
    assert rows[0]["n"] == 0


def test_approver_role_survives_the_resume_path(sandbox):
    """The role must reach the action, or the ceiling is un-approvable.

    `_normalise_resume` used to rebuild each decision and drop `role`, so
    `issue_partial_refund` always saw an empty role and rejected the refund
    even when the approver was a billing supervisor.
    """
    graph = build(sandbox, decide=_refund_decision_set())
    response = _invoke(graph, "Refund C-1001 $400 for the early cancellation", "S-ROLE")
    assert response.status == "awaiting_approval"

    finished = resume(
        response.session_id,
        [{
            "action": "issue_partial_refund",
            "approved": True,
            "approver": "dana",
            "role": "billing-supervisor",
            "reason": "pro-rated annual refund",
        }],
        settings=sandbox,
        graph=graph,
    )

    outcomes = {a.action: a for a in finished.actions}
    assert outcomes["issue_partial_refund"].ok, outcomes["issue_partial_refund"].error
    assert outcomes["issue_partial_refund"].result["over_ceiling"] is True
    assert outcomes["issue_partial_refund"].result["approver_role"] == "billing-supervisor"

    rows, _ = sql_store.execute_readonly(
        "SELECT amount_usd, approval_role FROM refunds", settings=sandbox
    )
    assert len(rows) == 1
    assert abs(rows[0]["amount_usd"] - 400.0) < 0.01
    assert rows[0]["approval_role"] == "billing-supervisor"


def test_refund_over_ceiling_is_refused_without_an_allowed_role(sandbox):
    """The ceiling must still bite when the approver holds the wrong role."""
    graph = build(sandbox, decide=_refund_decision_set())
    response = _invoke(graph, "Refund C-1001 $400 for the early cancellation", "S-NOROLE")

    finished = resume(
        response.session_id,
        [{
            "action": "issue_partial_refund",
            "approved": True,
            "approver": "dana",
            "role": "sales-rep",
            "reason": "pro-rated annual refund",
        }],
        settings=sandbox,
        graph=graph,
    )

    outcomes = {a.action: a for a in finished.actions}
    assert outcomes["issue_partial_refund"].ok is False
    assert "ceiling" in (outcomes["issue_partial_refund"].error or "").lower()

    rows, _ = sql_store.execute_readonly("SELECT COUNT(*) AS n FROM refunds", settings=sandbox)
    assert rows[0]["n"] == 0


def test_auto_approve_setting_skips_the_pause(sandbox, monkeypatch):
    monkeypatch.setenv("AUTO_APPROVE_IRREVERSIBLE", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    graph = build(settings, decide=_credit_decision_set())
    response = _invoke(graph, "Credit C-1001 $75 for the outage", "S-AUTO")

    assert response.status == "completed"
    outcomes = {a.action: a for a in response.actions}
    assert outcomes["apply_account_credit"].ok
    assert [a.status for a in response.approvals] == ["auto_approved"]


def test_pending_approval_is_queryable_across_processes(sandbox):
    from app import approval_store

    graph = build(sandbox, decide=_credit_decision_set())
    response = _invoke(graph, "Credit C-1001 $75 for the outage", "S-IDX")
    pending = approval_store.pending(sandbox)
    assert any(p["session_id"] == response.session_id for p in pending)


# --------------------------------------------------------------------------- #
# adversarial decisions
# --------------------------------------------------------------------------- #


def test_hallucinated_action_is_rejected_not_executed(sandbox):
    bad = DecisionSet(
        decisions=[
            Decision(
                id="evil",
                action="rm_rf_production",
                args={"target": "/"},
                why="I felt like it.",
            )
        ],
        answer_without_action="I could not do that.",
    )
    graph = build(sandbox, decide=bad)
    response = _invoke(graph, "delete everything", "S-HALLUC")
    assert response.actions == []
    rejected = [t for t in response.trace if t.data.get("rejected")]
    assert rejected
    assert "not a registered action" in rejected[0].data["rejected"][0]


def test_bad_arguments_are_rejected(sandbox):
    bad = DecisionSet(
        decisions=[
            Decision(
                id="bad",
                action="create_ticket",
                args={"customer_id": "not-an-id", "priority": "P1", "subject": "s"},
                why="x",
            )
        ]
    )
    graph = build(sandbox, decide=bad)
    response = _invoke(graph, "open a ticket", "S-BADARGS")
    assert response.actions == []
    rejected = [t for t in response.trace if t.data.get("rejected")]
    assert rejected and "customer_id" in rejected[0].data["rejected"][0]


def test_failing_action_is_recorded_as_failed(sandbox):
    failing = DecisionSet(
        decisions=[
            Decision(
                id="ghost",
                action="add_ticket_note",
                args={"ticket_id": "T-9999", "body": "x"},
                why="x",
            )
        ]
    )
    graph = build(sandbox, decide=failing)
    response = _invoke(graph, "note a ticket that does not exist", "S-FAIL")
    outcomes = {a.action: a for a in response.actions}
    assert outcomes["add_ticket_note"].ok is False
    assert "does not exist" in outcomes["add_ticket_note"].error


# --------------------------------------------------------------------------- #
# the reconcile and verify loops
# --------------------------------------------------------------------------- #


def test_insufficient_evidence_asks_the_user_instead_of_guessing(sandbox):
    graph = build(
        sandbox,
        reconcile=Reconciliation(
            status=ReconcileStatus.ASK_USER,
            reason="No record of which account this concerns.",
            user_question="Which customer id is this about?",
        ),
    )
    response = _invoke(graph, "cancel my subscription", "S-ASK")
    assert response.status == "needs_input"
    assert response.user_question == "Which customer id is this about?"
    assert response.actions == []


def test_gap_triggers_a_second_retrieval_pass(sandbox):
    counter = {"n": 0}

    def reconcile(schema, system, human, node):
        if schema is not Reconciliation:
            raise AssertionError(schema)
        counter["n"] += 1
        if counter["n"] == 1:
            return Reconciliation(
                status=ReconcileStatus.RE_RETRIEVE,
                reason="Missing the entitlement limit.",
                gaps=["storage allowance unknown"],
                follow_up_queries=["what is the storage allowance for the Growth plan"],
            )
        return Reconciliation(status=ReconcileStatus.PROCEED, reason="Gap closed.")

    from app.graph.builder import build_graph
    from app.llm import ScriptedLLM

    base = make_responder()
    llm = ScriptedLLM(
        lambda s, sy, h, n: reconcile(s, sy, h, n) if s is Reconciliation else base(s, sy, h, n),
        lambda sy, h, n: "ok",
        name="test",
    )
    graph = build_graph(sandbox, llm=llm, memory=get_memory_store(sandbox))
    response = _invoke(graph, "How much storage does C-1001 get?", "S-GAP")

    assert counter["n"] == 2
    assert any(t.node == "reconcile_pass2" for t in response.trace)
    assert any("storage" in (c.snippet or "").lower() for c in response.citations)


def test_verification_failure_causes_a_replan_then_succeeds(sandbox):
    calls = {"verify": 0, "plan": 0}

    def responder(schema, system, human, node):
        if schema is Verification:
            calls["verify"] += 1
            if calls["verify"] == 1:
                return Verification(
                    achieved=False,
                    confidence=0.3,
                    reasoning="The status change did not take effect.",
                    missing=["ticket status is still open"],
                    follow_up_queries=["current status of T-5002"],
                )
            return Verification(achieved=True, confidence=0.95, reasoning="Status now correct.")
        if schema is Plan:
            calls["plan"] += 1
        return make_responder()(schema, system, human, node)

    from app.graph.builder import build_graph
    from app.llm import ScriptedLLM

    llm = ScriptedLLM(responder, lambda sy, h, n: "answer", name="test")
    graph = build_graph(sandbox, llm=llm, memory=get_memory_store(sandbox))
    response = _invoke(graph, "close T-5002", "S-VERIFY")

    assert calls["verify"] == 2
    assert calls["plan"] == 2
    assert response.verification.achieved is True
    assert response.iterations == 2  # two verify passes: failed, then confirmed


def test_iteration_budget_is_respected(sandbox, monkeypatch):
    monkeypatch.setenv("MAX_ITERATIONS", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    always_unverified = Verification(
        achieved=False, confidence=0.1, reasoning="still not done", follow_up_queries=["more"]
    )
    from app.graph.builder import build_graph
    from app.llm import ScriptedLLM

    base = make_responder()
    llm = ScriptedLLM(
        lambda s, sy, h, n: always_unverified if s is Verification else base(s, sy, h, n),
        lambda sy, h, n: "answer",
        name="test",
    )
    graph = build_graph(settings, llm=llm, memory=get_memory_store(settings))
    response = _invoke(graph, "loop forever", "S-BUDGET")
    assert response.status == "completed"
    assert response.iterations <= settings.max_iterations


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #


def test_conversation_is_remembered_across_turns(sandbox):
    memory = get_memory_store(sandbox)
    graph = build(sandbox)
    session = "S-MEM-FIXED"
    first = run(
        ChatRequest(message="C-1001 needs a refund", session_id=session),
        settings=sandbox,
        graph=graph,
    )
    run(
        ChatRequest(message="T-5001 is the ticket", session_id=session),
        settings=sandbox,
        graph=graph,
    )

    items = memory.recall("turn", session_id=first.session_id, limit=10)
    assert items
    assert any("T-5001" in str(i.value) for i in items)


def test_long_term_fact_is_stored_per_customer(sandbox):
    memory = get_memory_store(sandbox)
    graph = build(sandbox)
    _invoke(graph, "C-1001 is unhappy", "S-LTM")
    facts = memory.recall("interaction", user_id="C-1001", limit=5)
    assert facts
    assert any("unhappy" in str(f.value) for f in facts)
