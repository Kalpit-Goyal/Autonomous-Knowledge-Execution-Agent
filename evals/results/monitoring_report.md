# Runtime monitoring report

Generated 2026-09-28T16:16:25+0530 from the SQLite `audit_log` table.

**Workload:** labeled scenario corpus, scripted LLM. Not production traffic.

## Audit volume

| Counter | Value |
| --- | --- |
| Total events | 298 |
| Distinct sessions | 29 |
| Failed events (`ok=0`) | 0 |
| Tokens in / out | 0 / 0 |
| Mean node duration | 0.0 ms |
| p50 / p95 / max | 0.0 / 0.0 / 0.0 ms |

## Events by type

| Event | Count |
| --- | --- |
| `reconcile` | 61 |
| `retrieve` | 59 |
| `plan` | 51 |
| `decide` | 43 |
| `verify` | 36 |
| `intake` | 30 |
| `respond` | 12 |
| `approval_requested` | 6 |

## Node execution frequency

| Node | Hits |
| --- | --- |
| `reconcile` | 61 |
| `retrieve` | 59 |
| `plan` | 51 |
| `decide` | 43 |
| `verify` | 36 |
| `intake` | 30 |
| `respond` | 12 |
| `gate_for_approval` | 6 |

## Latency caveat

The figures above are per-node durations from scripted runs, so they measure
orchestration overhead only. They are not a model-latency estimate. For the
real figure, the recorded live `gpt-oss-20b` turn held the connection for
221.87s and did not emit its first token until 221.51s, because every
reasoning node completes before `respond` runs. Per-node timing here is
therefore not a substitute for end-to-end measurement against the provider.

## Health checks

| Check | Status | Evidence |
| --- | --- | --- |
| no_failed_events | pass | 0 event(s) recorded ok=0 |
| actions_are_attributed | pass | every action and approval row names an actor |
| writes_follow_approval | pass | 0 action_start row(s) against 6 approval_requested and 0 auto-approved; none executed ungated |
| sessions_identified | pass | 29 distinct session id(s) in the trail |
| node_latency_sane | pass | p95 node duration 0ms across 298 samples |

**5/5 checks passing.**
