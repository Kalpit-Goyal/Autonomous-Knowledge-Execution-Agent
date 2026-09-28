# Retrieval evaluation (auto, k=1)

Queries: 22.  Gold labels in `evals/qrels.py`.

| Metric | All | Seeded conflicts | Ordinary |
| --- | --- | --- | --- |
| Precision@1 |  95.5% | 100.0% |  95.0% |
| Recall@1 |  90.9% |  75.0% |  92.5% |
| MRR | 0.955 | 1.000 | 0.950 |
| nDCG | 0.924 | 0.833 | 0.933 |
| Pass rate |  86.4% |  50.0% |  90.0% |

## Per query

| id | intent | P@k | R@k | rank | token | pass |
| --- | --- | --- | --- | --- | --- | --- |
| kb-01 | policy_lookup | 100.0% | 100.0% | 1 | y | PASS |
| kb-02 | seeded_conflict | 100.0% | 100.0% | 1 | y | PASS |
| kb-03 | seeded_conflict | 100.0% |  50.0% | 1 | y | FAIL |
| kb-04 | authorisation | 100.0% | 100.0% | 1 | y | PASS |
| kb-05 | definition | 100.0% | 100.0% | 1 | y | PASS |
| kb-06 | offboarding | 100.0% | 100.0% | 1 | y | PASS |
| kb-07 | retention | 100.0% | 100.0% | 1 | y | PASS |
| kb-08 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-09 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-10 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-11 | entitlement | 100.0% | 100.0% | 1 | y | PASS |
| kb-12 | sla | 100.0% | 100.0% | 1 | y | PASS |
| kb-13 | sla | 100.0% | 100.0% | 1 | y | PASS |
| kb-14 | sla | 100.0% | 100.0% | 1 | y | PASS |
| kb-15 | troubleshooting | 100.0% | 100.0% | 1 | y | PASS |
| kb-16 | troubleshooting | 100.0% | 100.0% | 1 | y | PASS |
| kb-17 | troubleshooting | 100.0% | 100.0% | 1 | y | PASS |
| kb-18 | troubleshooting | 100.0% | 100.0% | 1 | y | PASS |
| kb-19 | handoff | 100.0% |  50.0% | 1 | y | FAIL |
| kb-20 | policy_lookup | 100.0% | 100.0% | 1 | y | PASS |
| kb-21 | definition |   0.0% |   0.0% | - | N | FAIL |
| kb-22 | policy_lookup | 100.0% | 100.0% | 1 | y | PASS |

## Failures

- **kb-03** How much does the Growth plan cost per month? — gold ['billing-and-refunds.md', 'plans-and-entitlements.md'], got ['billing-and-refunds.md']
- **kb-19** When should I stop troubleshooting and escalate instead? — gold ['troubleshooting-playbook.md', 'sla-and-escalation.md'], got ['troubleshooting-playbook.md']
- **kb-21** What is a credit note and when do I get one? — gold ['billing-and-refunds.md'], got ['account-credits-and-goodwill.md']
