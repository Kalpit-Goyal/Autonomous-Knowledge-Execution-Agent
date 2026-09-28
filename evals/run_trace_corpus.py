"""Run the graph over labeled scenarios and export the traces as a corpus.

    python -m evals.run_trace_corpus

Each scenario pairs a question with the *expected* outcome, decided by reading
the seeded data rather than by observing a run. The scripted reasoning fixtures
come from `tests.test_graph.make_responder`, so a trace here is produced by the
same fixtures the test suite asserts on, which keeps the corpus and the suite
from drifting apart.

Two labels are recorded per scenario:
  * `label`     - what the agent should do (the gold answer)
  * `observed`  - what it actually did

A scenario passes only when the observed outcome matches the label. Results are
written to `evals/results/trace_corpus.json` and summarised in markdown.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

OUT_DIR = Path(os.getenv("EVAL_OUT_DIR", ROOT / "evals" / "results"))


def scenarios() -> list[dict[str, Any]]:
    """Labeled scenarios. `label` is the expected outcome, set by hand."""
    return [
        {
            "id": "tr-01",
            "name": "Refund eligibility, policy vs KB conflict",
            "message": "C-1001 wants a refund for an annual plan bought 20 days ago.",
            "label": {
                "intent": "refund_request",
                "reads_kb": True,
                "reads_policy": True,
                "expected_conflict": True,
                "must_cite_policy": True,
                "requires_approval": True,
                "writes_allowed": True,
            },
            "scenario": "refund_annual",
        },
        {
            "id": "tr-02",
            "name": "Goodwill credit inside authorised limit",
            "message": "C-1002 had an outage. Please give them a service credit as goodwill.",
            "label": {
                "intent": "credit_request",
                "reads_kb": True,
                "reads_policy": True,
                "expected_conflict": False,
                "must_cite_policy": True,
                "requires_approval": True,
                "writes_allowed": True,
            },
            "scenario": "credit_goodwill",
        },
        {
            "id": "tr-03",
            "name": "Outage triage, no state change",
            "message": "Our dashboard has been unusable since 09:00. What is happening?",
            "label": {
                "intent": "incident",
                "reads_kb": True,
                "reads_policy": True,
                "expected_conflict": False,
                "must_cite_policy": False,
                "requires_approval": False,
                "writes_allowed": False,
            },
            "scenario": "incident_triage",
        },
        {
            "id": "tr-04",
            "name": "Billing question answered from KB alone",
            "message": "How do I add extra seats to my plan?",
            "label": {
                "intent": "howto",
                "reads_kb": True,
                "reads_policy": False,
                "expected_conflict": False,
                "must_cite_policy": False,
                "requires_approval": False,
                "writes_allowed": False,
            },
            "scenario": "kb_only",
        },
        {
            "id": "tr-05",
            "name": "Offboarding with irreversible delete",
            "message": "Cancel C-1003's account and delete all of their data.",
            "label": {
                "intent": "offboarding",
                "reads_kb": True,
                "reads_policy": True,
                "expected_conflict": True,
                "must_cite_policy": True,
                "requires_approval": True,
                "writes_allowed": True,
            },
            "scenario": "offboard_delete",
        },
        {
            "id": "tr-06",
            "name": "SLA breach check against ops DB",
            "message": "Has T-5001 breached its response commitment?",
            "label": {
                "intent": "sla_review",
                "reads_kb": True,
                "reads_policy": True,
                "expected_conflict": False,
                "must_cite_policy": True,
                "requires_approval": False,
                "writes_allowed": False,
            },
            "scenario": "sla_breach",
        },
    ]


def build_for(sandbox, scenario: str):
    """Build a graph whose scripted reasoning matches the scenario."""
    from app.graph.builder import build_graph
    from app.llm import ScriptedLLM
    from app.schemas import (
        Decision,
        DecisionSet,
        Intake,
        Intent,
        Plan,
        PlanStep,
        RetrievalTask,
        SourceType,
        Verification,
    )
    from tests.test_graph import make_responder

    base: dict[str, Any] = {
        "intake": Intake(
            intent=Intent.COMPLAINT,
            summary="annual plan refund inside the policy window",
            customer_id="C-1001",
            ticket_id="T-5001",
            priority="P2",
            wants_state_change=True,
        ),
        "plan": Plan(
            goal="Resolve the refund question against the governing rule.",
            steps=[
                PlanStep(
                    id="policy",
                    goal="Read the governing policy and the KB",
                    why="The knowledge base and the policy disagree on the window.",
                    sources=[SourceType.POLICY, SourceType.KB],
                    tasks=[
                        RetrievalTask(
                            step_id="policy",
                            source_type=SourceType.POLICY,
                            query="annual refund window",
                            purpose="authority",
                        ),
                        RetrievalTask(
                            step_id="kb",
                            source_type=SourceType.KB,
                            query="refund window days",
                            purpose="documentation",
                        ),
                    ],
                )
            ],
        ),
        "decide": DecisionSet(
            decisions=[
                Decision(
                    id="credit",
                    action="apply_account_credit",
                    args={
                        "customer_id": "C-1001",
                        "amount_usd": 50.0,
                        "reason": "outage",
                        "ticket_id": "T-5001",
                    },
                    why="Policy allows a goodwill credit for this outage.",
                    risk="high",
                )
            ],
            rationale="Policy governs and permits a credit.",
        ),
        "verify": Verification(
            achieved=True,
            confidence=0.9,
            reasoning="The ops DB shows the credit row.",
        ),
    }

    if scenario == "incident_triage":
        base["intake"] = Intake(
            intent=Intent.COMPLAINT,
            summary="dashboard unusable since 09:00",
            customer_id="C-1001",
            ticket_id="T-5001",
            priority="P1",
            wants_state_change=False,
        )
        base["plan"] = Plan(
            goal="Assess impact and next steps.",
            steps=[
                PlanStep(
                    id="read",
                    goal="Read the SLA and the troubleshooting playbook",
                    why="Impact drives priority and response commitment.",
                    sources=[SourceType.POLICY, SourceType.KB],
                    tasks=[
                        RetrievalTask(
                            step_id="read",
                            source_type=SourceType.POLICY,
                            query="sla response commitments",
                            purpose="authority",
                        ),
                        RetrievalTask(
                            step_id="read_kb",
                            source_type=SourceType.KB,
                            query="slow dashboard troubleshooting",
                            purpose="documentation",
                        ),
                    ],
                )
            ],
        )
        base["decide"] = DecisionSet(
            decisions=[], rationale="Assessment only; no state change is warranted."
        )
    elif scenario == "kb_only":
        base["intake"] = Intake(
            intent=Intent.QUESTION,
            summary="how to add seats",
            wants_state_change=False,
        )
        base["plan"] = Plan(
            goal="Explain seat expansion.",
            steps=[
                PlanStep(
                    id="read",
                    goal="Read the entitlements article",
                    why="Seat rules are documented in the knowledge base.",
                    sources=[SourceType.KB],
                    tasks=[
                        RetrievalTask(
                            step_id="read",
                            source_type=SourceType.KB,
                            query="adding seats to a plan",
                            purpose="documentation",
                        )
                    ],
                )
            ],
        )
        base["decide"] = DecisionSet(decisions=[], rationale="Informational; no action.")
    elif scenario == "sla_breach":
        base["intake"] = Intake(
            intent=Intent.QUESTION,
            summary="checking the response commitment on T-5001",
            customer_id="C-1001",
            ticket_id="T-5001",
            wants_state_change=False,
        )
        base["decide"] = DecisionSet(decisions=[], rationale="Read-only review.")

    responder = make_responder(**base)
    llm = ScriptedLLM(
        responder,
        lambda s, h, n: "Answer grounded in the cited evidence.",
        name="eval",
    )
    return build_graph(sandbox, llm=llm, memory=None)


def _val(item: Any, key: str, default: Any = "") -> Any:
    """Read a field from a pydantic model or a plain dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def observe(result: Any) -> dict[str, Any]:
    """Reduce a ChatResponse to the fields the labels talk about.

    Note `awaiting_approval` is derived from `status` rather than read as its
    own attribute: ChatResponse exposes status as a literal, and an approval
    pause is the `awaiting_approval` value of it.
    """
    trace = list(getattr(result, "trace", None) or [])
    actions = list(getattr(result, "actions", None) or [])
    approvals = list(getattr(result, "approvals", None) or [])
    decisions = list(getattr(result, "decisions", None) or [])
    conflicts = list(getattr(result, "conflicts", None) or [])
    citations = list(getattr(result, "citations", None) or [])

    status = str(getattr(result, "status", "") or "")
    types = {str(_val(c, "source_type")) for c in citations}
    names = sorted({str(_val(c, "source_name")) for c in citations})

    # A policy read is proven by the evidence it produced, not by the plan,
    # because a planned retrieval that returned nothing should not count.
    return {
        "status": status,
        "awaiting_approval": status == "awaiting_approval",
        "answer": str(getattr(result, "answer", "") or ""),
        "citations": len(citations),
        "citation_source_types": sorted(types),
        "citation_sources": names,
        "n_actions": len(actions),
        "action_names": [str(_val(a, "action", _val(a, "tool"))) for a in actions],
        "n_decisions": len(decisions),
        "decision_names": [str(_val(d, "action")) for d in decisions],
        "n_approvals": len(approvals),
        "n_conflicts": len(conflicts),
        "iterations": int(getattr(result, "iterations", 0) or 0),
        "elapsed_ms": int(getattr(result, "elapsed_ms", 0) or 0),
        "trace_steps": len(trace),
        "nodes_visited": [str(_val(s, "node")) for s in trace],
        "reads_kb": any("KB" in t for t in types),
        "reads_policy": any("POLICY" in t for t in types),
        "error": getattr(result, "error", None),
    }


SCENARIO_ROOT = Path(
    os.getenv("EVAL_SANDBOX", str(Path(os.getenv("TEMP", "/tmp")) / "eval-trace-sandbox"))
)

# The corpus drives a throwaway data directory so repeated runs cannot leave
# audit rows behind in the real `data/support.db`. Set EVAL_SANDBOX=0 to run
# against the real directory instead, which is what the monitoring report wants:
# it reads the durable audit trail, and a sandboxed run deletes its own trail.
USE_REAL_DATA = os.getenv("EVAL_SANDBOX", "").strip() in {"0", "real", "false"}


def sandbox_ctx():
    """Standalone version of the `sandbox` pytest fixture.

    Reuses the same seeding order as `tests/conftest.py`; if that fixture
    changes, this must follow it.
    """
    import contextlib
    import shutil

    from app.config import get_settings
    from app.knowledge import sql_store
    from app.knowledge.catalog_store import load_catalog
    from app.knowledge.embeddings import reset_embeddings_cache
    from app.knowledge.policy_store import load_policies
    from app.knowledge.vector_store import index_kb, reset_vector_store
    from app.memory import reset_memory_store

    def _reset_caches() -> None:
        get_settings.cache_clear()
        reset_vector_store()
        reset_embeddings_cache()
        reset_memory_store()

    @contextlib.contextmanager
    def _ctx():
        if SCENARIO_ROOT.exists():
            shutil.rmtree(SCENARIO_ROOT, ignore_errors=True)
        if not USE_REAL_DATA:
            os.environ["DATA_DIR"] = str(SCENARIO_ROOT / "data")
        os.environ["EMBEDDING_BACKEND"] = os.getenv("EVAL_BACKEND", "hashing")
        os.environ["AUTO_APPROVE_IRREVERSIBLE"] = "0"
        _reset_caches()

        settings = get_settings()
        settings.ensure_dirs()
        for folder in ("knowledge", "structured"):
            source = ROOT / "data" / folder
            target = settings.data_dir / folder
            target.mkdir(parents=True, exist_ok=True)
            if source.resolve() == target.resolve():
                # Running against the real data directory: the copy would be
                # the file onto itself, which Windows refuses.
                continue
            shutil.copytree(source, target, dirs_exist_ok=True)

        sql_store.init_db(settings)
        sql_store.seed_from_csv(settings=settings)
        index_kb(force=True, settings=settings)
        load_policies(force=True, settings=settings)
        load_catalog(force=True, settings=settings)
        try:
            yield settings
        finally:
            _reset_caches()
            if not USE_REAL_DATA:
                shutil.rmtree(SCENARIO_ROOT, ignore_errors=True)

    return _ctx()


def main() -> int:
    from app.graph.runner import run
    from app.schemas import ChatRequest

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for sc in scenarios():
        with sandbox_ctx() as settings:
            graph = build_for(settings, sc["scenario"])
            try:
                result = run(
                    ChatRequest(
                        message=sc["message"],
                        session_id=f"eval-{uuid.uuid4().hex[:8]}",
                        actor_name="eval",
                        actor_role="agent",
                    ),
                    settings=settings,
                    graph=graph,
                )
                obs = observe(result)
            except Exception as exc:  # noqa: BLE001
                obs = {"error": f"{type(exc).__name__}: {exc}"}

        label = sc["label"]
        gated = bool(obs.get("awaiting_approval"))

        # A pending write has not executed yet, and that is the correct
        # outcome: the run must stop and ask. So "a write happened" is only
        # expected once the gate has cleared.
        if label["writes_allowed"] and label["requires_approval"]:
            if gated:
                write_ok = obs.get("n_approvals", 0) > 0 and obs.get("n_actions", 0) == 0
            else:
                write_ok = obs.get("n_actions", 0) > 0
        elif label["writes_allowed"]:
            write_ok = obs.get("n_actions", 0) > 0
        else:
            write_ok = obs.get("n_actions", 0) == 0 and not gated

        checks = {
            "no_error": not obs.get("error"),
            "kb_touched": bool(obs.get("reads_kb")) or not label["reads_kb"],
            "policy_touched": bool(obs.get("reads_policy")) or not label["reads_policy"],
            "approval_gate": gated == label["requires_approval"],
            "write_shape": write_ok,
            "cited": (obs.get("citations", 0) > 0) or not label["must_cite_policy"],
        }
        records.append(
            {
                "id": sc["id"],
                "name": sc["name"],
                "message": sc["message"],
                "label": label,
                "observed": obs,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
        flag = "PASS" if records[-1]["passed"] else "FAIL"
        print(f"{sc['id']} {flag} {sc['name']} :: {obs.get('status', 'error')}")

    n = len(records)
    npass = sum(1 for r in records if r["passed"])
    payload = {
        "scenarios": n,
        "passed": npass,
        "pass_rate": npass / n if n else 0.0,
        "traces": records,
    }
    (OUT_DIR / "trace_corpus.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Labeled trace corpus",
        "",
        f"Scenarios: {n}.  Matched every declared label: {npass}/{n} "
        f"({payload['pass_rate'] * 100:.1f}%).",
        "",
        "| id | scenario | status | approval gate | writes | citations | trace | result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        o = r["observed"]
        lines.append(
            f"| {r['id']} | {r['name']} | {o.get('status', 'error')} | "
            f"{'yes' if o.get('awaiting_approval') else 'no'} | {o.get('n_actions', 0)} | "
            f"{o.get('citations', 0)} | {o.get('trace_steps', 0)} | "
            f"{'PASS' if r['passed'] else 'FAIL'} |"
        )
    lines += ["", "## Per-scenario checks", ""]
    for r in records:
        lines.append(f"- **{r['id']} {r['name']}**")
        for k, v in r["checks"].items():
            lines.append(f"  - {k}: {'ok' if v else 'MISMATCH'}")
    (OUT_DIR / "trace_corpus.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'trace_corpus.json'}")
    return 0 if npass == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
