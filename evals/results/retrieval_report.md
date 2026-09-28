# Retrieval evaluation (auto, k=5)

Queries: 22.  Gold labels in `evals/qrels.py`.

| Metric | All | Seeded conflicts | Ordinary |
| --- | --- | --- | --- |
| Precision@5 |  62.9% | 100.0% |  59.2% |
| Recall@5 | 100.0% | 100.0% | 100.0% |
| MRR | 0.977 | 1.000 | 0.975 |
| nDCG | 0.977 | 1.000 | 0.975 |
| Pass rate | 100.0% | 100.0% | 100.0% |

## Per query

| id | intent | P@k | R@k | rank | token | pass |
| --- | --- | --- | --- | --- | --- | --- |
| kb-01 | policy_lookup |  50.0% | 100.0% | 1 | y | PASS |
| kb-02 | seeded_conflict | 100.0% | 100.0% | 1 | y | PASS |
| kb-03 | seeded_conflict | 100.0% | 100.0% | 1 | y | PASS |
| kb-04 | authorisation |  50.0% | 100.0% | 1 | y | PASS |
| kb-05 | definition |  50.0% | 100.0% | 1 | y | PASS |
| kb-06 | offboarding |  50.0% | 100.0% | 1 | y | PASS |
| kb-07 | retention |  50.0% | 100.0% | 1 | y | PASS |
| kb-08 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-09 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-10 | entitlement |  33.3% | 100.0% | 1 | y | PASS |
| kb-11 | entitlement |  50.0% | 100.0% | 1 | y | PASS |
| kb-12 | sla |  50.0% | 100.0% | 1 | y | PASS |
| kb-13 | sla | 100.0% | 100.0% | 1 | y | PASS |
| kb-14 | sla |  33.3% | 100.0% | 1 | y | PASS |
| kb-15 | troubleshooting |  50.0% | 100.0% | 1 | y | PASS |
| kb-16 | troubleshooting |  33.3% | 100.0% | 1 | y | PASS |
| kb-17 | troubleshooting |  33.3% | 100.0% | 1 | y | PASS |
| kb-18 | troubleshooting |  50.0% | 100.0% | 1 | y | PASS |
| kb-19 | handoff | 100.0% | 100.0% | 1 | y | PASS |
| kb-20 | policy_lookup | 100.0% | 100.0% | 1 | y | PASS |
| kb-21 | definition |  50.0% | 100.0% | 2 | y | PASS |
| kb-22 | policy_lookup |  50.0% | 100.0% | 1 | y | PASS |
