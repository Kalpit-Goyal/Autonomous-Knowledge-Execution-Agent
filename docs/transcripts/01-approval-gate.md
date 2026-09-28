# Approval gate transcript

Captured 2026-09-28T16:29:26+0530 against `the in-process graph (evals/run_trace_corpus fixtures)`.

> Scripted reasoning, real graph, real ops database, real approval interrupt. The provider is stubbed so the turn is instant and reproducible; everything shown about the gate, the evidence and the trace is genuine.

## tr-01 — Refund eligibility, policy vs KB conflict

```
$ POST /chat   (actor: t-dana, role: support_lead)
  message: "C-1001 wants a refund for an annual plan bought 20 days ago."

HTTP 200
  status            awaiting_approval
  elapsed_ms        60
  iterations        0
  trace steps       6
  citations         10  (SourceType.KB, SourceType.POLICY)
  conflicts found   1
  decisions         1  ['apply_account_credit']
  approvals raised  1
  actions executed  0
```

**Node order**

```
intake -> plan -> retrieve -> reconcile -> decide -> gate_for_approval
```

**Answer as returned**

```
I have stopped short of changing anything. These actions need a human decision:

- apply_account_credit: 'apply_account_credit' is marked irreversible in the action registry, so a human must authorise it before it runs.
  why: Policy allows a goodwill credit for this outage.

Approve or reject them in the approval panel and I will continue from here.
```

**Pending approval**

```
{
  "id": "apr_78ffad6b871a",
  "action": "apply_account_credit",
  "risk": "RiskLevel.HIGH",
  "reason": "",
  "status": "pending"
}
```

**Conflict reported to the user**

```
{
  "summary": "Refund window: policy 14 days vs knowledge base 30 days",
  "iteration": 0
}
```

## tr-04 — Billing question answered from KB alone

```
$ POST /chat   (actor: t-dana, role: support_lead)
  message: "How do I add extra seats to my plan?"

HTTP 200
  status            completed
  elapsed_ms        71
  iterations        1
  trace steps       8
  citations         5  (SourceType.KB)
  conflicts found   1
  decisions         0  []
  approvals raised  0
  actions executed  0
```

**Node order**

```
intake -> plan -> retrieve -> reconcile -> decide -> execute -> verify -> respond
```

**Answer as returned**

```
Answer grounded in the cited evidence.
```

**Conflict reported to the user**

```
{
  "summary": "Refund window: policy 14 days vs knowledge base 30 days",
  "iteration": 0
}
```

## tr-06 — SLA breach check against ops DB

```
$ POST /chat   (actor: t-dana, role: support_lead)
  message: "Has T-5001 breached its response commitment?"

HTTP 200
  status            completed
  elapsed_ms        72
  iterations        1
  trace steps       8
  citations         10  (SourceType.KB, SourceType.POLICY)
  conflicts found   1
  decisions         0  []
  approvals raised  0
  actions executed  0
```

**Node order**

```
intake -> plan -> retrieve -> reconcile -> decide -> execute -> verify -> respond
```

**Answer as returned**

```
Answer grounded in the cited evidence.
```

**Conflict reported to the user**

```
{
  "summary": "Refund window: policy 14 days vs knowledge base 30 days",
  "iteration": 0
}
```

## Reading these

- `awaiting_approval` with `actions executed 0` is the correct outcome, not a failure. The gate exists so an irreversible write cannot happen without a human.
- The scripted stub returns the same prose for every read-only turn. Only the gate, the evidence and the trace are interesting here; the wording is a fixture.
- Every scenario reports a conflict because the scripted reconciliation fixture declares one. In a live run the count reflects real disagreements found in the evidence.
