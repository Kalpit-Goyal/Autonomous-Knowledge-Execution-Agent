# Sample run transcripts

Verbatim captures of the agent doing work. Every file here was written by
`scripts/capture_transcripts.py`, which performs the request and formats the
actual response. Nothing is hand-typed, so a transcript cannot drift from what
the service did.

| File | What it shows | Provider |
| --- | --- | --- |
| `01-approval-gate.md` | Three labelled scenarios, including the irreversible-write gate halting with `actions executed 0` | scripted |
| `02-document-management.md` | Add, list, reject and remove a knowledge base document over HTTP | none (local) |
| `03-streaming.md` | SSE headers, frame order, and the in-band error path | real, rate-limited |
| `04-live-agent-turn.md` | A full successful turn: node timeline, token stream, citations, conflict resolution | real, hand-transcribed |
| `05-retrieval-eval.md` | Precision/recall against labelled queries, real embedding backend | n/a (offline) |

## Reproducing them

```bash
# the API has to be up for the two HTTP transcripts
python -m uvicorn app.api.main:app --port 8000

python scripts/capture_transcripts.py                    # 01, 02, 03
python scripts/capture_transcripts.py --only corpus       # 01 only, no API needed
python scripts/capture_transcripts.py --live              # adds 04, spends tokens
```

`--live` is a real `gpt-oss-20b` call and takes minutes. It is the only
transcript that depends on a working key.

## Which files are machine-generated

`01`, `02`, `03` and `05` are written by the capture script or the eval
harness from a live request. `04` is the exception: it is transcribed by hand
from a real earlier run, because the account hit its daily Groq quota
partway through collection and a fresh run now fails with a 429. Its header
says so. The script refuses to overwrite it with an empty result rather than
destroy a good capture.

## Reading 01 next to 04

They look like different systems. 01 finishes in ~80ms with a scripted
responder; 04 takes over three minutes with the real model, replans four times,
and only then emits an answer.

The gate behaves identically in both — `awaiting_approval`, zero actions
executed — because the interrupt is in the graph, not in the model. That is the
part worth taking from these transcripts: the safety property does not depend on
the provider behaving.

## Two things to know before quoting these

**The scripted transcripts stub the prose.** In 01 the read-only turns all
return "Answer grounded in the cited evidence." Only the gate, the evidence
list and the trace are meaningful there.

**Provider errors are redacted on the way in.** `03-streaming.md` contains a
real 429 quoting an organisation id and a billing URL. The capture script masks
both before writing, so the committed file is safe to share.
