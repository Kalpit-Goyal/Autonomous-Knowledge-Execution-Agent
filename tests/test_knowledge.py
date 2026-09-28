"""The knowledge layer: chunking, the four sources, and the SQL guard."""

from __future__ import annotations

import pytest

from app.knowledge import catalog_store, policy_store, sql_store
from app.knowledge.chunking import chunk_markdown, stable_chunk_id
from app.knowledge.vector_store import collection_stats, ensure_indexed, search

# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def test_chunk_markdown_splits_on_headings_and_keeps_context():
    text = """# Title

Intro paragraph.

## Section A

Body A.

## Section B

Body B.
"""
    chunks = chunk_markdown(
        text, source_name="doc.md", chunk_size=400, chunk_overlap=40
    )
    assert len(chunks) >= 3
    headings = [c.heading for c in chunks]
    assert any("Section A" in h for h in headings)
    assert any("Section B" in h for h in headings)
    assert all(c.text.strip() for c in chunks)


def test_chunk_ids_are_stable_across_runs():
    text = "# A\n\nalpha beta gamma\n\n## B\n\ndelta epsilon\n"
    first = [
        stable_chunk_id("doc.md", c)
        for c in chunk_markdown(text, source_name="doc.md", chunk_size=400, chunk_overlap=40)
    ]
    second = [
        stable_chunk_id("doc.md", c)
        for c in chunk_markdown(text, source_name="doc.md", chunk_size=400, chunk_overlap=40)
    ]
    assert first == second
    assert len(set(first)) == len(first)


def test_long_body_is_split_rather_than_truncated():
    body = "sentence number %d. " * 1
    text = "# Long\n\n" + " ".join(body % i for i in range(400))
    chunks = chunk_markdown(
        text, source_name="long.md", chunk_size=300, chunk_overlap=30
    )
    assert len(chunks) > 1


# --------------------------------------------------------------------------- #
# the four sources
# --------------------------------------------------------------------------- #


def test_kb_is_indexed_and_searchable(sandbox):
    stats = collection_stats(sandbox)
    assert stats["documents"] > 0
    hits = search("refund window for an annual plan", top_k=3, settings=sandbox)
    assert hits
    assert all(h.source_type.value == "kb" for h in hits)
    assert any("refund" in h.citation.lower() or "refund" in h.snippet.lower() for h in hits)


def test_ensure_indexed_is_idempotent(sandbox):
    first = ensure_indexed(sandbox)
    count = collection_stats(sandbox)["documents"]
    ensure_indexed(sandbox)
    ensure_indexed(sandbox)
    assert collection_stats(sandbox)["documents"] == count
    assert first["documents"] == count


def test_policy_lookup_returns_the_authoritative_window(sandbox):
    hits = policy_store.policy_evidence("annual plan refund window", top_k=5)
    assert hits
    text = " ".join(h.snippet for h in hits)
    assert "14" in text
    assert hits[0].source_type.value == "policy"


def test_catalog_lookup_returns_prices(sandbox):
    hits = catalog_store.catalog_evidence("what does Growth cost per month", top_k=4)
    assert hits
    text = " ".join(h.snippet for h in hits)
    assert "499" in text


def test_ops_db_query_returns_rows(sandbox):
    rows, statement = sql_store.execute_readonly(
        "SELECT customer_id, plan FROM customers WHERE health = 'at_risk'", settings=sandbox
    )
    assert rows
    assert "LIMIT" in statement.upper()
    assert all(set(r) == {"customer_id", "plan"} for r in rows)


def test_ops_db_and_policy_disagree_on_refund_window(sandbox):
    """The seeded conflict must actually be present, or reconciliation is theatre."""
    policy_text = " ".join(
        h.snippet for h in policy_store.policy_evidence("annual refund window", top_k=5)
    )
    kb_text = " ".join(
        h.snippet for h in search("refund window days", top_k=5, settings=sandbox)
    )
    assert "14" in policy_text
    assert "30" in kb_text


# --------------------------------------------------------------------------- #
# the SQL guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM tickets",
        "UPDATE customers SET health = 'ok'",
        "INSERT INTO tickets (ticket_id) VALUES ('T-1')",
        "DROP TABLE tickets",
        "ALTER TABLE tickets ADD COLUMN x TEXT",
        "PRAGMA table_info(customers)",
    ],
)
def test_guard_rejects_writes(sql, sandbox):
    ok, reason = sql_store.validate_sql(sql)
    assert not ok
    assert reason


def test_guard_rejects_multiple_statements(sandbox):
    ok, reason = sql_store.validate_sql("SELECT 1; DROP TABLE tickets")
    assert not ok
    assert "single statement" in reason


def test_guard_rejects_empty_and_comment_leads(sandbox):
    assert not sql_store.validate_sql("")[0]
    assert not sql_store.validate_sql("-- sneaky\nSELECT 1")[0]


def test_guard_adds_a_limit(sandbox):
    ok, statement = sql_store.validate_sql("SELECT * FROM customers")
    assert ok
    assert "LIMIT" in statement.upper()


def test_execute_readonly_never_mutates(sandbox):
    before, _ = sql_store.execute_readonly("SELECT COUNT(*) AS n FROM tickets", settings=sandbox)
    sql_store.execute_readonly("DELETE FROM tickets", settings=sandbox)
    after, _ = sql_store.execute_readonly("SELECT COUNT(*) AS n FROM tickets", settings=sandbox)
    assert before == after


def test_model_visible_schema_hides_internal_tables(sandbox):
    schema = sql_store.schema_description(sandbox)
    assert "customers" in schema
    assert "audit_log" not in schema
    assert "approvals" not in schema
    assert "messages" not in schema


def test_model_can_see_every_table_an_action_writes_to(sandbox):
    """If the agent cannot query a table, it cannot verify its own write.

    `refunds` was added for issue_partial_refund but left out of
    MODEL_VISIBLE_TABLES, so a refund was unconfirmable by SQL.
    """
    schema = sql_store.schema_description(sandbox)
    for table in ("account_credits", "refunds", "ticket_events"):
        assert table in schema, f"{table} is written by an action but hidden from the model"

    # Every model-visible table must actually exist in the DDL, so the schema
    # summary can never advertise a table the read tool cannot query.
    with sql_store._connect(sandbox) as conn:
        for table in sql_store.MODEL_VISIBLE_TABLES:
            found = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            assert found, f"MODEL_VISIBLE_TABLES names {table} but it is not in the DDL"


def test_seed_populated_every_table(sandbox):
    counts = sql_store.row_counts(sandbox)
    assert counts["customers"] == 10
    assert counts["subscriptions"] == 10
    assert counts["tickets"] == 12
    assert counts["account_credits"] == 0


def test_reset_also_clears_agent_written_rows(sandbox):
    """`seed --reset` must give a clean slate, not just restore the CSVs."""
    with sql_store._connect(sandbox) as conn:
        conn.execute(
            "INSERT INTO account_credits "
            "(credit_id, customer_id, amount_usd, reason, created_at) "
            "VALUES ('CR-X', 'C-1001', 5.0, 'test', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO audit_log (event_type, reasoning) VALUES ('intake', 'test')"
        )
    assert sql_store.row_counts(sandbox)["account_credits"] == 1

    sql_store.seed_from_csv(reset=True, settings=sandbox)

    counts = sql_store.row_counts(sandbox)
    assert counts["account_credits"] == 0
    assert counts["audit_log"] == 0
    assert counts["customers"] == 10, "reference rows must be restored"
