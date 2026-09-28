# Live agent turn

> **Provenance:** transcribed by hand from a real `openai/gpt-oss-20b` run
> captured earlier in the working session against `http://127.0.0.1:8000`. The
> numbers below are as recorded at the time (wall clock, first-token offset,
> frame counts, node timeline, status, citation and trace counts, and the
> answer text).
>
> **It is not machine-generated.** The capture script normally writes this file
> verbatim, but the account hit its 200k tokens-per-day quota partway through
> collecting transcripts, so a fresh run now returns 429 before the graph does
> any work — see `03-streaming.md`. The script refuses to overwrite this file
> with an empty result rather than destroy a good capture, which means this
> copy is manual. Everything in it came from a real run; it just was not
> re-captured by the tool.

## Request

```
$ curl -N -X POST localhost:8000/chat/stream \
    -H 'content-type: application/json' \
    -d '{"message": "In one short sentence, what is the refund window for annual plans?"}'
```

```
HTTP 200
content-type: text/event-stream; charset=utf-8
cache-control: no-cache, no-transform
x-accel-buffering: no
```

## Node timeline

Reasoning progress, as emitted:

```
t=   1.38s  intake
t=   1.88s  plan
t=   1.89s  retrieve
t=   2.85s  reconcile
t=   3.79s  decide
t=   3.79s  gate_for_approval
t=   3.79s  execute
t=   4.50s  verify
t=   4.50s  after_verify
t=  46.20s  decide          <- replan 1
t= 119.40s  decide          <- replan 2
t= 194.10s  decide          <- replan 3
t= 221.51s  first token
t= 221.86s  respond
t= 221.87s  done
```

## Answer

```
The refund window for annual plans is 14 days. The policy states "Annual
subscriptions are non-refundable after 14 days" — this overrides the standard
30-day window for any plan with billing_cycle = annual, which supersedes the KB
claim of a 30-day window.

Action taken: looked up the business policy and cross-checked the knowledge
base; identified and noted the conflict, then applied the governing policy
rule.

Action not taken: no credit, refund, or other state change was applied, because
the question asked what the window is, not for a refund to be issued.
```

`ev_…` markers appear inline in the raw output. Those are Groq's own citation
artifacts, not something the agent emits.

## Summary

```
wall clock        221.87s
first token       221.51s
token frames      99
node events       26
final status      completed
citations         13
trace steps       26
iterations        4
```

## The thing this transcript is actually about

**The first token arrives at 99.8% of the way through the turn.** Everything
before it is reasoning: 26 node events, four replans, no output.

So the streaming work did not make this turn feel fast. It made the last 0.36
seconds visible. What actually made the 221 seconds tolerable was the `node`
frames — a user watching this sees progress at 1.38s, 46s, 119s and 194s instead
of a spinner. Had the stream carried reasoning deltas as well as answer deltas,
the turn would have felt substantially different.

This is a design consequence, not a bug: `respond` is the only node that
streams, because the others emit structured objects with no sensible
token-level rendering. Widening the stream to cover reasoning would mean
streaming partial JSON into the UI, which is a real decision rather than an
obvious improvement.

## Why it replanned four times

`MAX_ITERATIONS=4`, and the run used all four. The `decide` node at 46s, 119s
and 194s is that node re-entering after `verify` reported the goal unmet. Each
cycle is a full re-plan and re-retrieval, not a cheap retry.

At 221s per turn against a 200k tokens/day quota, the arithmetic is roughly
4–5 such turns per day on the free tier. That is the practical ceiling on this
configuration, and it is worth stating plainly rather than describing streaming
as though it solved throughput.

## Conflict handling

This question targets one of the two seeded disagreements
(`policies.json: conflicts_on_purpose`):

- `kb/billing-and-refunds.md` — 30-day refund window for all plans
- `policies.json: refunds.annual_plan_window_days` — 14 for annual plans

The answer applied the policy value and said so explicitly, naming the
overridden KB claim and why. The KB is designed to lose this argument: the
article itself defers to the policy and catalog for authoritative figures. An
agent that quietly returned 30 would read as more confident and be more wrong.
