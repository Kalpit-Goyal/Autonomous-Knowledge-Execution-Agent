# Retrieval evaluation (hashing, k=5)

Queries: 22.  Gold labels in `evals/qrels.py`.

| Metric | All | Seeded conflicts | Ordinary |
| --- | --- | --- | --- |
| Precision@5 |  34.8% |  41.7% |  34.2% |
| Recall@5 |  95.5% | 100.0% |  95.0% |
| MRR | 0.879 | 1.000 | 0.867 |
| nDCG | 0.879 | 1.000 | 0.867 |
| Pass rate |  95.5% | 100.0% |  95.0% |

## Per query

| id | intent | P@k | R@k | rank | token | pass |
| --- | --- | --- | --- | --- | --- | --- |
| kb-01 | policy_lookup |  33.3% | 100.0% | 1 | y | PASS |
| kb-02 | seeded_conflict |  33.3% | 100.0% | 1 | y | PASS |
| kb-03 | seeded_conflict |  50.0% | 100.0% | 1 | y | PASS |
| kb-04 | authorisation |  33.3% | 100.0% | 3 | y | PASS |
| kb-05 | definition |  50.0% | 100.0% | 1 | y | PASS |
| kb-06 | offboarding |  33.3% | 100.0% | 1 | y | PASS |
| kb-07 | retention |  33.3% | 100.0% | 1 | y | PASS |
| kb-08 | entitlement |  33.3% | 100.0% | 2 | y | PASS |
| kb-09 | entitlement |  50.0% | 100.0% | 1 | y | PASS |
| kb-10 | entitlement |   0.0% |   0.0% | - | N | FAIL |
| kb-11 | entitlement |  33.3% | 100.0% | 1 | y | PASS |
| kb-12 | sla |  25.0% | 100.0% | 1 | y | PASS |
| kb-13 | sla |  33.3% | 100.0% | 1 | y | PASS |
| kb-14 | sla |  33.3% | 100.0% | 1 | y | PASS |
| kb-15 | troubleshooting |  33.3% | 100.0% | 1 | y | PASS |
| kb-16 | troubleshooting |  25.0% | 100.0% | 1 | y | PASS |
| kb-17 | troubleshooting |  33.3% | 100.0% | 1 | y | PASS |
| kb-18 | troubleshooting |  33.3% | 100.0% | 2 | y | PASS |
| kb-19 | handoff |  66.7% | 100.0% | 1 | y | PASS |
| kb-20 | policy_lookup |  25.0% | 100.0% | 1 | y | PASS |
| kb-21 | definition |  25.0% | 100.0% | 1 | y | PASS |
| kb-22 | policy_lookup |  50.0% | 100.0% | 1 | y | PASS |

## Failures

- **kb-10** How long is the free trial? — gold ['plans-and-entitlements.md'], got ['troubleshooting-playbook.md', 'data-export-and-offboarding.md', 'billing-and-refunds.md', 'account-credits-and-goodwill.md']
