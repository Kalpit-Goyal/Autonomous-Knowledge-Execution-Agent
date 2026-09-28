"""Labeled retrieval evaluation for the knowledge base.

Every label here is authored by reading the seeded article it refers to, then
written down *before* running retrieval. That ordering is the whole point: the
gold set is not the retrieval output, so it can actually disagree with it.

Each case declares the documents that must be retrieved (`sources`) and the
literal token the answer must be able to lean on (`must_contain`). The token
check exists because document-level relevance is coarse - the agent quotes a
specific line, not a file.

Cases that touch a seeded disagreement carry `expect_conflict`, and are scored
separately. The KB is wrong on those values by design, so scoring them as
plain relevance would reward the agent for retrieving a passage it must then
override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Case:
    qid: str
    question: str
    sources: tuple[str, ...]
    must_contain: str
    intent: str
    expect_conflict: bool = False
    note: str = ""
    label_revised: str = ""


CASES: tuple[Case, ...] = (
    Case(
        "kb-01",
        "How long do I have to request a refund on a monthly plan?",
        ("billing-and-refunds.md",),
        "30",
        "policy_lookup",
    ),
    Case(
        "kb-02",
        "What is the refund window for an annual subscription?",
        ("billing-and-refunds.md",),
        "30",
        "seeded_conflict",
        expect_conflict=True,
        note="KB says 30 days; policies.json says 14 for annual. The KB value "
        "is the wrong one and must be overridden.",
    ),
    Case(
        "kb-03",
        "How much does the Growth plan cost per month?",
        ("billing-and-refunds.md", "plans-and-entitlements.md"),
        "450",
        "seeded_conflict",
        expect_conflict=True,
        note="KB pricing table says 450; catalog.csv says 499, which governs.",
        label_revised="first labelled must_contain='499'. That is the catalog value, "
        "and the catalog is a structured source, so the token can never appear in "
        "KB text. The KB is supposed to return 450 so the agent has something to "
        "override. Corrected to '450' after the first run.",
    ),
    Case(
        "kb-04",
        "Who is allowed to authorise an account credit?",
        ("account-credits-and-goodwill.md",),
        "credit",
        "authorisation",
    ),
    Case(
        "kb-05",
        "What is the difference between a goodwill credit and a refund?",
        ("account-credits-and-goodwill.md",),
        "goodwill",
        "definition",
    ),
    Case(
        "kb-06",
        "What happens to my data when I cancel?",
        ("data-export-and-offboarding.md",),
        "export",
        "offboarding",
    ),
    Case(
        "kb-07",
        "How long do I have to request a data export?",
        ("data-export-and-offboarding.md",),
        "export",
        "retention",
    ),
    Case(
        "kb-08",
        "How many seats are included in the Scale plan?",
        ("plans-and-entitlements.md",),
        "seat",
        "entitlement",
    ),
    Case(
        "kb-09",
        "How do I add more seats to my plan?",
        ("plans-and-entitlements.md",),
        "seat",
        "entitlement",
    ),
    Case(
        "kb-10",
        "How long is the free trial?",
        ("plans-and-entitlements.md",),
        "trial",
        "entitlement",
    ),
    Case(
        "kb-11",
        "What usage limits apply to the Starter plan?",
        ("plans-and-entitlements.md",),
        "Starter",
        "entitlement",
    ),
    Case(
        "kb-12",
        "What counts as a Sev-1 incident?",
        ("sla-and-escalation.md",),
        "P1",
        "sla",
        label_revised="first labelled must_contain='Sev-1'. The article never uses that "
        "term; it uses 'P1 Critical'. Retrieval was already correct at rank 1 and only "
        "the token assertion was wrong. Corrected to 'P1' after the first run.",
    ),
    Case(
        "kb-13",
        "When does the response clock stop during an incident?",
        ("sla-and-escalation.md",),
        "clock",
        "sla",
    ),
    Case(
        "kb-14",
        "How do I report a security incident?",
        ("sla-and-escalation.md",),
        "security",
        "sla",
    ),
    Case(
        "kb-15",
        "My customers cannot log in after a password reset. What do I try?",
        ("troubleshooting-playbook.md",),
        "login",
        "troubleshooting",
    ),
    Case(
        "kb-16",
        "My webhooks stopped delivering. What is the fix?",
        ("troubleshooting-playbook.md",),
        "webhook",
        "troubleshooting",
    ),
    Case(
        "kb-17",
        "Dashboards are loading very slowly. What should I check?",
        ("troubleshooting-playbook.md",),
        "dashboard",
        "troubleshooting",
    ),
    Case(
        "kb-18",
        "I see duplicate records after a data sync. How do I clean up?",
        ("troubleshooting-playbook.md",),
        "duplicate",
        "troubleshooting",
    ),
    Case(
        "kb-19",
        "When should I stop troubleshooting and escalate instead?",
        ("troubleshooting-playbook.md", "sla-and-escalation.md"),
        "escalat",
        "handoff",
    ),
    Case(
        "kb-20",
        "Are partial refunds allowed on any plan?",
        ("billing-and-refunds.md",),
        "refund",
        "policy_lookup",
    ),
    Case(
        "kb-21",
        "What is a credit note and when do I get one?",
        ("billing-and-refunds.md",),
        "credit note",
        "definition",
    ),
    Case(
        "kb-22",
        "Which items can never be refunded?",
        ("billing-and-refunds.md",),
        "non-refundable",
        "policy_lookup",
    ),
)


@dataclass
class Result:
    case: Case
    retrieved: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    hit_rank: int | None = None
    token_found: bool = False
    precision_at_k: float = 0.0
    recall: float = 0.0
    reciprocal_rank: float = 0.0
    dcg: float = 0.0
    idcg: float = 0.0
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def grade(
    case: Case, retrieved: list[str], snippets: list[str], scores: list[float], k: int
) -> Result:
    """Score one case.

    `retrieved` is the ordered list of distinct source names, best first.
    Precision uses the distinct-document prefix actually consumed, because
    asking "of the 5 passages I sent the model, how many were relevant" is the
    number that governs context budget, and a document contributing three
    passages should not be counted three times as if it were three facts.
    """
    res = Result(case=case, retrieved=retrieved, snippets=snippets, scores=scores)

    window = retrieved[:k]
    relevant = set(case.sources)
    res.precision_at_k = len([d for d in window if d in relevant]) / len(window) if window else 0.0
    res.recall = len([d for d in window if d in relevant]) / len(relevant)

    for idx, doc in enumerate(retrieved):
        if doc in relevant:
            res.hit_rank = idx + 1
            res.reciprocal_rank = 1.0 / (idx + 1)
            break

    joined = "\n".join(snippets).lower()
    res.token_found = case.must_contain.lower() in joined

    gains = [1 if d in relevant else 0 for d in retrieved]
    res.dcg = sum(g / (i + 1) for i, g in enumerate(gains))
    ideal = [1] * len(relevant) + [0] * (len(retrieved) - len(relevant))
    res.idcg = sum(1.0 / (i + 1) for i, g in enumerate(ideal) if g)

    # Pass requires the gold document inside the window AND the answer token
    # being present somewhere in what was retrieved.
    res.passed = bool(window) and res.precision_at_k > 0 and res.recall == 1.0 and res.token_found
    return res


def macro(results: list[Result]) -> dict[str, float]:
    if not results:
        return {}
    n = len(results)

    def mean(fn) -> float:
        return sum(fn(r) for r in results) / n

    ndcg = mean(lambda r: (r.dcg / r.idcg) if r.idcg else 0.0)
    return {
        "n": n,
        "precision_at_k": mean(lambda r: r.precision_at_k),
        "recall_at_k": mean(lambda r: r.recall),
        "mrr": mean(lambda r: r.reciprocal_rank),
        "ndcg": ndcg,
        "pass_rate": mean(lambda r: 1.0 if r.passed else 0.0),
    }
