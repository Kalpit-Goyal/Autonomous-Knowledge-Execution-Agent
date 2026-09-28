# Retrieval evaluation transcript

Captured from `python -m evals.run_retrieval_eval --top-k 5`, real embedding
backend (ONNX MiniLM), 22 labelled queries. No provider call: this measures
which document the retriever puts in front of the model.

## The question

> In one short sentence, what is the refund window for annual plans?

This is a deliberately bad question for a naive retriever. The knowledge base
says 30 days for all plans. The policy says 14 for annual. Both answers are in
the corpus, only one is correct, and the wrong one is the one a keyword search
finds more easily.

## k-sweep

```
$ foreach ($k in 1,3,5,8) { python -m evals.run_retrieval_eval --top-k $k }
```

| k | Precision@k | Recall@k | MRR | nDCG | Pass |
| --- | --- | --- | --- | --- | --- |
| 1 | 95.5% | 90.9% | 0.955 | 0.924 | 86.4% |
| 3 | 78.8% | 95.5% | 0.955 | 0.955 | 95.5% |
| 5 | 62.9% | 100.0% | 0.977 | 0.977 | 100.0% |
| 8 | 40.2% | 100.0% | 0.977 | 0.977 | 100.0% |

Configured value is `RETRIEVAL_TOP_K=5`.

Recall saturates at 5 and precision keeps degrading past it, so k=5 buys the
last 4.5% of recall for 16 points of precision. k=3 is the better trade if
context budget matters. That curve is the argument for the configured value
being a measurement rather than a guess.

## The same queries without a real embedding model

```
$ python -m evals.run_retrieval_eval --top-k 5 --backend hashing
P@5= 34.8% R@5= 95.5% MRR=0.879 nDCG=0.879 pass= 95.5%
```

| Backend | P@5 | R@5 | MRR | Pass |
| --- | --- | --- | --- | --- |
| ONNX MiniLM | 62.9% | 100.0% | 0.977 | 100.0% |
| hashing (offline) | 34.8% | 95.5% | 0.879 | 95.5% |

Precision nearly halves. Recall holds up, which is the misleading part: a suite
that only asserted "the right document is somewhere in the top 5" would pass on
the hashing backend and tell you nothing.

This is why the offline test suite asserts index membership and vector cleanup,
never ranking quality. Retrieval quality is only measurable against the real
embedding backend, which is what this report is for.

## Two labels were wrong

Both were caught by the token check, and both were the label's fault, not
retrieval's. The revisions are recorded in `evals/qrels.py` under
`label_revised` rather than corrected silently.

**kb-03** — labelled `must_contain="499"`. That is the catalog price, and the
catalog is a structured source, so the token cannot appear in KB text. The KB
is *supposed* to return 450, because that is the value the agent has to
override with 499. Retrieval was correct; the label demanded the impossible.

**kb-12** — labelled `must_contain="Sev-1"`. The article uses `P1 Critical`.
The document was already at rank 1, so only the token assertion failed.

Revising labels after seeing output is the standard way an IR evaluation
manufactures a good score, so both the revisions and the pre-revision numbers
are on the record.

Pre-revision: P@5 62.9%, pass 90.9% (19/22). Post-revision: pass 100% (22/22).
Precision was unaffected by either change — the tokens never entered the
ranking, only the answer-token assertion.

## Seeded conflicts, scored separately

Two queries target disagreements planted in `policies.json` under
`conflicts_on_purpose`. Scoring them as plain relevance would reward the agent
for retrieving a passage it is required to override, so they are broken out:

| Slice | n | P@5 | R@5 | MRR | Pass |
| --- | --- | --- | --- | --- | --- |
| Ordinary queries | 20 | 59.2% | 100.0% | 0.975 | 100.0% |
| Seeded conflicts | 2 | 100.0% | 100.0% | 1.000 | 100.0% |

Both slices retrieve perfectly. The conflict slice scores *better* than the
ordinary one, which is worth pausing on: those two queries have both documents
in the gold set (`billing-and-refunds.md` **and** `catalog.csv`), so finding
both is a pass. Scoring a conflict as plain relevance would reward retrieving
a passage the agent is then required to override, which is why the slice is
reported separately at all.

The real conflict handling is downstream. In `01-approval-gate.md` the turn
surfaces `Refund window: policy 14 days vs knowledge base 30 days` and stops
for approval — that is the reconcile node applying the authority hierarchy, and
retrieval metrics cannot see it.

## Per-intent

| Intent | n | P@5 | R@5 | MRR |
| --- | --- | --- | --- | --- |
| seeded_conflict | 2 | 100.0% | 100.0% | 1.000 |
| handoff | 1 | 100.0% | 100.0% | 1.000 |
| entitlement | 4 | 70.8% | 100.0% | 1.000 |
| policy_lookup | 3 | 66.7% | 100.0% | 1.000 |
| sla | 3 | 61.1% | 100.0% | 1.000 |
| authorisation | 1 | 50.0% | 100.0% | 1.000 |
| definition | 2 | 50.0% | 100.0% | 0.750 |
| offboarding | 1 | 50.0% | 100.0% | 1.000 |
| retention | 1 | 50.0% | 100.0% | 1.000 |
| troubleshooting | 4 | 41.7% | 100.0% | 1.000 |

MRR is 1.000 for every intent except `definition`, so the gold document is
rank 1 essentially everywhere. All the precision variation is trailing
irrelevant chunks, not mis-ranked relevant ones — which is the benign failure
mode. A retriever that buried the right document at rank 4 would show the
opposite profile: high recall, low MRR.

`troubleshooting` is the noisiest (41.7%): "dashboard is loading slowly",
"webhooks stopped delivering" and the handoff query all pull the playbook plus
neighbouring passages, because one article covers five distinct symptoms.

## What this does not measure

- **End-to-end answer correctness.** This scores retrieval only. Whether the
  agent then uses the passage correctly is the trace corpus's job, and it is
  also where the scripted fixtures limit what can be claimed.
- **Robustness to paraphrase.** 22 queries written by the same hand that wrote
  the gold documents. Real query sets drift from document wording, and this one
  will flatter the retriever.
- **The ops database, policy file and catalog.** All gold labels are KB
  documents, because that is where the seeded text lives.
- **Multi-turn degradation.** Each query is independent. A session that
  progressively narrows to one document is untested.
