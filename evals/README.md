# Evaluation artifacts

Three runnable harnesses, each writing both JSON and a readable Markdown
report to `evals/results/`.

| Script | Produces | Question it answers |
| --- | --- | --- |
| `run_retrieval_eval.py` | `retrieval_report.*`, `retrieval_k{1,3,5,8}.*`, `retrieval_hashing.*` | Does retrieval put the right document in front of the model, and how much noise comes with it? |
| `run_trace_corpus.py` | `trace_corpus.*` | Does the agent take the labeled path end to end, including stopping for approval? |
| `run_monitoring.py` | `monitoring_report.*` | What does the system record about its own behaviour at runtime? |

## Running them

```bash
# 1. retrieval, with the real embedding backend
python -m evals.run_retrieval_eval --top-k 5

# k-sweep, to see where precision and recall cross
foreach ($k in 1,3,5,8) { python -m evals.run_retrieval_eval --top-k $k }

# same queries against the offline hashing backend, for comparison
python -m evals.run_retrieval_eval --top-k 5 --backend hashing

# 2. labeled trace corpus
#    sandboxed (default): audit rows are thrown away afterwards
python -m evals.run_trace_corpus
#    against the real data dir: leaves a durable audit trail for step 3
$env:EVAL_SANDBOX="0"; python -m evals.run_trace_corpus

# 3. runtime report, read back out of the audit trail
python -m evals.run_monitoring
```

Step 2 must be run with `EVAL_SANDBOX=0` before step 3, or the audit rows the
report reads will have been deleted along with the sandbox.

## What the labels are, and where they come from

`qrels.py` holds 22 retrieval queries. Each declares the documents that must be
retrieved plus a literal token the retrieved text must contain. Document
relevance alone is too coarse, because the agent quotes a specific line rather
than a file.

`run_trace_corpus.py` holds 6 end-to-end scenarios. Each declares the expected
outcome — whether the KB and policy should be read, whether an approval gate
should fire, whether a write should happen — and passes only if the observed
response matches on every declared field.

Both label sets were written by reading the seeded data, not by watching a run
produce an answer. The corpus reuses `tests.test_graph.make_responder`, the same
scripted fixtures the test suite asserts against, so the two cannot drift apart
without the fixtures changing under both.

**Two labels were corrected after the first run**, and the corrections are
recorded in the `label_revised` field of each case rather than quietly applied:

- `kb-03` was labelled `must_contain="499"`, which is the catalog value. The
  catalog is a structured source, so that token can never appear in KB text.
  Retrieval was never wrong here; the label was.
- `kb-12` was labelled `must_contain="Sev-1"`. The article uses `P1 Critical`.
  The document was already at rank 1; only the token assertion failed.

Revising a label after seeing output is the main way an IR evaluation talks
itself into a good score, which is why both revisions are in the file and the
pre-revision numbers are in the write-up.

## Results as recorded

Retrieval, real backend (ONNX MiniLM), 22 queries:

| k | Precision@k | Recall@k | MRR | nDCG | Pass |
| --- | --- | --- | --- | --- | --- |
| 1 | 95.5% | 90.9% | 0.955 | 0.924 | 86.4% |
| 3 | 78.8% | 95.5% | 0.955 | 0.955 | 95.5% |
| 5 | 62.9% | 100.0% | 0.977 | 0.977 | 100.0% |
| 8 | 40.2% | 100.0% | 0.977 | 0.977 | 100.0% |

Same queries, hashing backend: P@5 34.8%, R@5 95.5%, MRR 0.879, pass 95.5%.
The gap is the reason retrieval quality is not asserted in the offline suite.

Trace corpus: 6/6 scenarios matching every declared label.
