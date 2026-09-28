# Retrieval evaluation (auto, k=3)

Queries: 22.  Gold labels in `evals/qrels.py`.

| Metric | All | Seeded conflicts | Ordinary |
| --- | --- | --- | --- |
| Precision@3 |  78.8% | 100.0% |  76.7% |
| Recall@3 |  95.5% | 100.0% |  95.0% |
| MRR | 0.955 | 1.000 | 0.950 |
| nDCG | 0.955 | 1.000 | 0.950 |
| Pass rate |  95.5% | 100.0% |  95.0% |

## Per query

| id | intent | P@k | R@k | rank | token | pass |
| --- | --- | --- | --- | --- | --- | --- |
| kb-01 | policy_lookup | 100.0% | 100.0% | 1 | y | PASS |
| kb-02 | seeded_conflict | 100.0% | 100.0% | 1 | y | PASS |
| kb-03 | seeded_conflict | 100.0% | 100.0% | 1 | y | PASS |
| kb-04 | authorisation | 100.0% | 100.0% | 1 | y | PASS |
| kb-05 | definition | 100.0% | 100.0% | 1 | y | PASS |
| kb-06 | offboarding | 100.0% | 100.0% | 1 | y | PASS |
| kb-07 | retention | 100.0% | 100.0% | 1 | y | PASS |
| kb-08 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-09 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-10 | entitlement |  50.0% | 100.0% | 1 | y | PASS |
| kb-11 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-12 | sla |  50.0% | 100.0% | 1 | y | PASS |
| kb-13 | sla | 100.0% | 100.0% | 1 | y | PASS |
| kb-14 | sla |  50.0% | 100.0% | 1 | y | PASS |
| kb-15 | troubleshooting | 100.0% | 100.0% | 1 | y | PASS |
| kb-16 | troubleshooting |  50.0% | 100.0% | 1 | y | PASS |
| kb-17 | troubleshooting |  33.3% | 100.0% | 1 | y | PASS |
| kb-18 | troubleshooting |  50.0% | 100.0% | 1 | y | PASS |
| kb-19 | handoff | 100.0% | 100.0% | 1 | y | PASS |
| kb-20 | policy_lookup | 100.0% | 100.0% | 1 | y | PASS |
| kb-21 | definition |   0.0% |   0.0% | - | N | FAIL |
| kb-22 | policy_lookup |  50.0% | 100.0% | 1 | y | PASS |

## Failures

- **kb-21** What is a credit note and when do I get one? — gold ['billing-and-refunds.md'], got ['account-credits-and-goodwill.md']
