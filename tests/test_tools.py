"""Read tools, the action registry, and the handlers' real side effects."""

from __future__ import annotations

import json

import pytest

from app.knowledge import policy_store, sql_store
from app.schemas import SourceType
from app.tools import (
    ACTIONS,
    READ_TOOLS,
    get_action,
    is_irreversible,
    needs_approval,
    registry_as_prompt_block,
    registry_as_tool_schemas,
    run_read_tool,
    validate_args,
)
from app.tools.action_registry import ActionContext
from app.tools.read_tools import tool_for_source


@pytest.fixture
def ctx(sandbox):
    return ActionContext(session_id="T-SESSION", actor="tester")


# --------------------------------------------------------------------------- #
# read tools
# --------------------------------------------------------------------------- #


def test_every_source_type_maps_to_a_registered_tool():
    for source in SourceType:
        assert tool_for_source(source) in READ_TOOLS


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("search_knowledge_base", {"query": "rotate an API key"}),
        ("lookup_business_policy", {"question": "P1 escalation routing"}),
        ("lookup_product_catalog", {"question": "Scale plan price"}),
        ("describe_operations_schema", {}),
        (
            "query_operations_data",
            {
                "question": "open tickets",
                "sql": "SELECT ticket_id FROM tickets WHERE status != 'closed' LIMIT 3",
            },
        ),
    ],
)
def test_read_tools_return_evidence(name, kwargs, sandbox):
    results = run_read_tool(name, **kwargs)
    assert results
    assert all(r.citation for r in results)


def test_unknown_read_tool_reports_instead_of_raising(sandbox):
    results = run_read_tool("no_such_tool", query="x")
    assert len(results) == 1
    assert "unknown read tool" in results[0].snippet


def test_rejected_sql_yields_no_evidence_rather_than_an_error(sandbox):
    """A rejected statement is not an exception; it simply produces nothing."""
    assert run_read_tool(
        "query_operations_data", question="x", sql="DROP TABLE tickets"
    ) == []


def test_read_tool_exception_is_captured_not_raised(sandbox, monkeypatch):
    from app.tools import read_tools

    def boom(**kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setitem(read_tools.READ_TOOLS["lookup_business_policy"], "fn", boom)
    results = run_read_tool("lookup_business_policy", question="x")
    assert len(results) == 1
    assert "failed" in results[0].citation
    assert "backend exploded" in results[0].snippet


def test_ops_tool_returns_nothing_for_no_rows(sandbox):
    assert run_read_tool(
        "query_operations_data", question="x", sql="SELECT * FROM customers WHERE 1=0"
    ) == []


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_catalogue_and_tool_schemas_stay_in_step():
    assert len(registry_as_tool_schemas()) == len(ACTIONS)
    block = registry_as_prompt_block()
    for name in ACTIONS:
        assert name in block


def test_catalogue_marks_the_dangerous_actions():
    block = registry_as_prompt_block()
    assert "IRREVERSIBLE" in block
    assert "apply_account_credit" in block
    assert "cancel_subscription" in block


def test_only_irreversible_actions_need_approval():
    for name, spec in ACTIONS.items():
        expected = spec.irreversible
        assert is_irreversible(name) is expected
        assert needs_approval(name) is expected, name


def test_policy_can_require_approval_for_a_reversible_action(sandbox):
    """Approval rules are data, so a policy can tighten one without a code change."""
    from app.knowledge import policy_store

    path = sandbox.policies_path
    policies = json.loads(path.read_text(encoding="utf-8"))
    policies["approvals"]["requires_human_approval_actions"].append("schedule_callback")
    path.write_text(json.dumps(policies, indent=2), encoding="utf-8")
    policy_store.load_policies(force=True, settings=sandbox)
    try:
        assert needs_approval("schedule_callback") is True
    finally:
        path.write_text(json.dumps(policies, indent=2), encoding="utf-8")
        restored = json.loads(path.read_text(encoding="utf-8"))
        restored["approvals"]["requires_human_approval_actions"] = [
            a
            for a in restored["approvals"]["requires_human_approval_actions"]
            if a != "schedule_callback"
        ]
        path.write_text(json.dumps(restored, indent=2), encoding="utf-8")
        policy_store.load_policies(force=True, settings=sandbox)


# --------------------------------------------------------------------------- #
# argument validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "action,args",
    [
        ("create_ticket", {"customer_id": "C-1001", "subject": "s", "priority": "P1"}),
        ("add_ticket_note", {"ticket_id": "T-5001", "body": "b"}),
        ("update_ticket_status", {"ticket_id": "T-5001", "status": "resolved"}),
        ("escalate_ticket", {"ticket_id": "T-5001", "reason": "r"}),
        ("send_customer_reply", {"customer_id": "C-1001", "subject": "s", "body": "b"}),
        (
            "schedule_callback",
            {"customer_id": "C-1001", "when": "2030-01-01T10:00:00Z", "purpose": "p"},
        ),
        ("export_case_report", {"customer_id": "C-1001"}),
        ("apply_account_credit", {"customer_id": "C-1001", "amount_usd": 10, "reason": "r"}),
        ("cancel_subscription", {"subscription_id": "S-9001", "reason": "r"}),
    ],
)
def test_valid_arguments_are_accepted(action, args, sandbox):
    spec = get_action(action)
    cleaned, error = validate_args(spec, args)
    assert error is None
    assert cleaned is not None


@pytest.mark.parametrize(
    "action,args",
    [
        ("create_ticket", {"customer_id": "nope", "subject": "s", "priority": "P1"}),
        ("create_ticket", {"customer_id": "C-1001", "subject": "s", "priority": "P9"}),
        ("create_ticket", {"customer_id": "C-1001"}),
        ("update_ticket_status", {"ticket_id": "T-5001", "status": "exploded"}),
        ("add_ticket_note", {"ticket_id": "bad", "body": "b"}),
        ("apply_account_credit", {"customer_id": "C-1001", "amount_usd": -5, "reason": "r"}),
        ("apply_account_credit", {"customer_id": "C-1001", "reason": "r"}),
        (
            "schedule_callback",
            {"customer_id": "C-1001", "when": "not-a-date", "purpose": "p"},
        ),
        ("cancel_subscription", {"subscription_id": "nope", "reason": "r"}),
    ],
)
def test_invalid_arguments_are_rejected_with_a_message(action, args, sandbox):
    cleaned, error = validate_args(get_action(action), args)
    assert cleaned is None
    assert error and action in error


# --------------------------------------------------------------------------- #
# handlers really do something
# --------------------------------------------------------------------------- #


def test_create_ticket_inserts_a_row(sandbox, ctx):
    spec = get_action("create_ticket")
    args, _ = validate_args(spec, {"customer_id": "C-1001", "subject": "s", "priority": "P2"})
    result = spec.fn(args, ctx)
    rows, _ = sql_store.execute_readonly(
        f"SELECT subject FROM tickets WHERE ticket_id = '{result['ticket_id']}'", settings=sandbox
    )
    assert rows and rows[0]["subject"] == "s"


def test_add_ticket_note_appends_an_event(sandbox, ctx):
    before, _ = sql_store.execute_readonly(
        "SELECT COUNT(*) AS n FROM ticket_events", settings=sandbox
    )
    spec = get_action("add_ticket_note")
    args, _ = validate_args(spec, {"ticket_id": "T-5001", "body": "looked into it"})
    spec.fn(args, ctx)
    after, _ = sql_store.execute_readonly(
        "SELECT COUNT(*) AS n FROM ticket_events", settings=sandbox
    )
    assert after[0]["n"] == before[0]["n"] + 1


def test_update_ticket_status_records_the_transition(sandbox, ctx):
    spec = get_action("update_ticket_status")
    args, _ = validate_args(
        spec, {"ticket_id": "T-5002", "status": "awaiting_customer_response", "reason": "asked"}
    )
    result = spec.fn(args, ctx)
    assert result["previous_status"] != "awaiting_customer_response"
    rows, _ = sql_store.execute_readonly(
        "SELECT status FROM tickets WHERE ticket_id = 'T-5002'", settings=sandbox
    )
    assert rows[0]["status"] == "awaiting_customer_response"


def test_escalate_writes_a_notice_and_raises_priority(sandbox, ctx):
    spec = get_action("escalate_ticket")
    args, _ = validate_args(
        spec, {"ticket_id": "T-5001", "reason": "security incident", "severity": "P1"}
    )
    result = spec.fn(args, ctx)
    assert result["priority"] == "P1"
    assert result["notified"]
    assert (sandbox.outbox_dir / result["outbox_file"]).exists()


def test_send_reply_lands_in_the_outbox(sandbox, ctx):
    spec = get_action("send_customer_reply")
    args, _ = validate_args(
        spec,
        {
            "customer_id": "C-1001",
            "subject": "hi",
            "body": "we are on it",
            "ticket_id": "T-5001",
        },
    )
    result = spec.fn(args, ctx)
    path = sandbox.outbox_dir / result["outbox_file"]
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "we are on it" in text
    assert "tester" in text or "on it" in text


def test_credit_and_cancellation_really_mutate(sandbox, ctx):
    credit = get_action("apply_account_credit")
    args, _ = validate_args(
        credit, {"customer_id": "C-1001", "amount_usd": 42.5, "reason": "outage"}
    )
    credit.fn(args, ctx)

    rows, _ = sql_store.execute_readonly(
        "SELECT amount_usd FROM account_credits", settings=sandbox
    )
    assert len(rows) == 1
    assert abs(rows[0]["amount_usd"] - 42.5) < 0.01

    sub_id = _first_subscription(sandbox)
    cancel = get_action("cancel_subscription")
    args, _ = validate_args(cancel, {"subscription_id": sub_id, "reason": "churn"})
    outcome = cancel.fn(args, ctx)

    rows, _ = sql_store.execute_readonly(
        f"SELECT status FROM subscriptions WHERE subscription_id = '{sub_id}'", settings=sandbox
    )
    assert rows[0]["status"] == "cancelled_at_period_end"
    assert outcome["previous_status"] != "cancelled_at_period_end"
    assert any(sandbox.outbox_dir.glob(f"cancellation_{sub_id}_*.json"))


def test_handlers_reject_unknown_ids(sandbox, ctx):
    for action, args in [
        ("add_ticket_note", {"ticket_id": "T-9999", "body": "b"}),
        ("escalate_ticket", {"ticket_id": "T-9999", "reason": "r"}),
        ("apply_account_credit", {"customer_id": "C-9999", "amount_usd": 5, "reason": "r"}),
        ("cancel_subscription", {"subscription_id": "S-9999", "reason": "r"}),
        ("issue_partial_refund", {"customer_id": "C-9999", "amount_usd": 5, "reason": "r"}),
    ]:
        spec = get_action(action)
        cleaned, _ = validate_args(spec, args)
        with pytest.raises(ValueError):
            spec.fn(cleaned, ctx)


def test_refund_under_the_ceiling_needs_no_special_role(sandbox, ctx):
    """The ceiling comes from policies.json, and roles only gate amounts above it."""
    spec = get_action("issue_partial_refund")
    args, _ = validate_args(
        spec, {"customer_id": "C-1001", "amount_usd": 120.0, "reason": "partial month"}
    )
    result = spec.fn(args, ctx)

    ceiling = float(
        policy_store.get_policy("refunds").get("partial_refund_ceiling_usd", 0)
    )
    assert result["ceiling_usd"] == ceiling
    assert result["over_ceiling"] is False

    rows, _ = sql_store.execute_readonly(
        "SELECT amount_usd, reason FROM refunds", settings=sandbox
    )
    assert len(rows) == 1
    assert abs(rows[0]["amount_usd"] - 120.0) < 0.01


def test_refund_above_the_ceiling_demands_a_policy_approver_role(sandbox, ctx):
    spec = get_action("issue_partial_refund")
    args, _ = validate_args(
        spec, {"customer_id": "C-1001", "amount_usd": 900.0, "reason": "annual downgrades"}
    )

    with pytest.raises(PermissionError) as excinfo:
        spec.fn(args, ctx)
    assert "ceiling" in str(excinfo.value)

    rows, _ = sql_store.execute_readonly("SELECT COUNT(*) AS n FROM refunds", settings=sandbox)
    assert rows[0]["n"] == 0, "a refused refund must not write a row"

    ctx.approver_role = "billing-supervisor"
    result = spec.fn(args, ctx)
    assert result["over_ceiling"] is True


def test_refund_ceiling_comes_from_policy_not_the_handler(sandbox, ctx):
    """Changing the policy file changes the threshold with no code change."""
    spec = get_action("issue_partial_refund")
    args, _ = validate_args(
        spec, {"customer_id": "C-1001", "amount_usd": 300.0, "reason": "test"}
    )
    with pytest.raises(PermissionError):
        spec.fn(args, ctx)

    path = sandbox.structured_dir / "policies.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["refunds"]["partial_refund_ceiling_usd"] = 500
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    policy_store.load_policies(force=True)

    result = spec.fn(args, ctx)
    assert result["ceiling_usd"] == 500.0
    assert result["over_ceiling"] is False


def test_export_report_contains_the_ticket_history(sandbox, ctx):
    spec = get_action("export_case_report")
    args, _ = validate_args(
        spec, {"customer_id": "C-1001", "format": "json", "include": ["tickets", "subscription"]}
    )
    result = spec.fn(args, ctx)
    payload = json.loads((sandbox.outbox_dir / result["outbox_file"]).read_text(encoding="utf-8"))
    assert payload["sections"]["tickets"]
    assert payload["sections"]["subscription"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _first_subscription(sandbox) -> str:
    rows, _ = sql_store.execute_readonly(
        "SELECT subscription_id FROM subscriptions LIMIT 1", settings=sandbox
    )
    return rows[0]["subscription_id"]
