"""Run the labeled retrieval eval and write a report.

    python -m evals.run_retrieval_eval

Reports the *real* embedding backend by default. The offline test suite forces
the deterministic hashing backend, whose scores are near-uniform and carry no
semantic signal, so measuring precision/recall against it would be measuring
noise. Pass --backend hashing to see that for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = Path(os.getenv("EVAL_OUT_DIR", ROOT / "evals" / "results"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.getenv("EMBEDDING_BACKEND", "auto"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="retrieval_report.json")
    ap.add_argument("--markdown", default="retrieval_report.md")
    args = ap.parse_args()

    os.environ["EMBEDDING_BACKEND"] = args.backend

    from app.config import get_settings
    from app.knowledge.vector_store import ensure_indexed, search
    from evals.qrels import CASES, grade, macro

    settings = get_settings()
    index = ensure_indexed(settings=settings)
    backend = index.get("backend", settings.embedding_backend)

    results = []
    for case in CASES:
        hits = search(case.question, top_k=args.top_k, settings=settings)
        retrieved: list[str] = []
        snippets: list[str] = []
        scores: list[float] = []
        for h in hits:
            if h.source_name not in retrieved:
                retrieved.append(h.source_name)
            snippets.append(h.snippet)
            scores.append(h.score)
        results.append(grade(case, retrieved, snippets, scores, args.top_k))

    overall = macro(results)
    conflict = macro([r for r in results if r.case.expect_conflict])
    plain = macro([r for r in results if not r.case.expect_conflict])
    by_intent: dict[str, dict[str, float]] = {}
    for intent in sorted({r.case.intent for r in results}):
        by_intent[intent] = macro([r for r in results if r.case.intent == intent])

    report = {
        "backend": backend,
        "top_k": args.top_k,
        "indexed_documents": index.get("documents"),
        "overall": overall,
        "seeded_conflict_cases": conflict,
        "ordinary_cases": plain,
        "by_intent": by_intent,
        "cases": [
            {
                "qid": r.case.qid,
                "question": r.case.question,
                "intent": r.case.intent,
                "expect_conflict": r.case.expect_conflict,
                "gold_sources": list(r.case.sources),
                "retrieved": r.retrieved[: args.top_k],
                "hit_rank": r.hit_rank,
                "precision_at_k": round(r.precision_at_k, 4),
                "recall": round(r.recall, 4),
                "reciprocal_rank": round(r.reciprocal_rank, 4),
                "ndcg": round((r.dcg / r.idcg) if r.idcg else 0.0, 4),
                "token_found": r.token_found,
                "passed": r.passed,
                "note": r.case.note,
            }
            for r in results
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    lines = [
        f"# Retrieval evaluation ({backend}, k={args.top_k})",
        "",
        f"Queries: {overall['n']}.  Gold labels in `evals/qrels.py`.",
        "",
        "| Metric | All | Seeded conflicts | Ordinary |",
        "| --- | --- | --- | --- |",
        f"| Precision@{args.top_k} | {pct(overall['precision_at_k'])} | "
        f"{pct(conflict['precision_at_k'])} | {pct(plain['precision_at_k'])} |",
        f"| Recall@{args.top_k} | {pct(overall['recall_at_k'])} | "
        f"{pct(conflict['recall_at_k'])} | {pct(plain['recall_at_k'])} |",
        f"| MRR | {overall['mrr']:.3f} | {conflict['mrr']:.3f} | {plain['mrr']:.3f} |",
        f"| nDCG | {overall['ndcg']:.3f} | {conflict['ndcg']:.3f} | {plain['ndcg']:.3f} |",
        f"| Pass rate | {pct(overall['pass_rate'])} | {pct(conflict['pass_rate'])} | "
        f"{pct(plain['pass_rate'])} |",
        "",
        "## Per query",
        "",
        "| id | intent | P@k | R@k | rank | token | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.case.qid} | {r.case.intent} | {pct(r.precision_at_k)} | "
            f"{pct(r.recall)} | {r.hit_rank or '-'} | "
            f"{'y' if r.token_found else 'N'} | {'PASS' if r.passed else 'FAIL'} |"
        )
    failed = [r for r in results if not r.passed]
    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines.append(
                f"- **{r.case.qid}** {r.case.question} — gold {list(r.case.sources)}, "
                f"got {r.retrieved[: args.top_k]}"
            )
    (OUT_DIR / args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"backend={backend} k={args.top_k} queries={overall['n']}")
    print(
        f"P@{args.top_k}={pct(overall['precision_at_k'])} "
        f"R@{args.top_k}={pct(overall['recall_at_k'])} "
        f"MRR={overall['mrr']:.3f} nDCG={overall['ndcg']:.3f} "
        f"pass={pct(overall['pass_rate'])}"
    )
    print(f"wrote {OUT_DIR / args.out} and {OUT_DIR / args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
