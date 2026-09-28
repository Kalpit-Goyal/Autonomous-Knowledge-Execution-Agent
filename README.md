# Autonomous Knowledge Execution Agent

A support agent for **Meridian Cloud** that answers questions from internal sources, reasons over
what it finds, and takes real actions on the operations database — pausing for human approval
before anything irreversible.

The company, the data, and the policies are all fictional. Every business rule lives in
`data/structured/policies.json` or `data/knowledge/kb/*.md`, not in the code, so the agent has to
read and reconcile them at runtime instead of reciting a script.

**Stack:** LangGraph · Groq (`openai/gpt-oss-20b`) · Chroma · SQLite · FastAPI · Streamlit.

---

## What it actually does

Ask a question in plain language and the agent:

1. reads the request and pulls out ids, intent and entities,
2. decides which internal sources it needs and queries them in parallel,
3. **reconciles disagreements between those sources** instead of picking one silently,
4. picks actions from a registered catalogue and explains why, with citations,
5. **stops and asks a human** before any irreversible action,
6. executes the approved actions as real SQLite writes,
7. verifies afterwards that the goal was met, and re-plans if it was not.

```
                       ┌──────────── replan (budget: MAX_ITERATIONS) ──────────┐
                       │                                                       │
  intake ──► plan ──► retrieve ──► reconcile ──► decide ──► gate_for_approval ──┴─► execute ──► verify ──► respond
                        ▲          │           │            │                       ▲                     │
                        └── pass 2 ┘           │            └── run (reversible)   │                     │
                                   │           │                                 │                     │
                                   └── ask_user┘            collect_approval ───────┘                     │
                                                              (LangGraph interrupt)                      │
                                                                                                          │
                                                        respond ◄────────────── retry while unverified ──┘
```

Every hop is checkpointed to SQLite, so a pending approval survives a restart of the API process.

---

## Quick start

Requires **Python 3.12** on Windows, macOS or Linux.

```bash
git clone <your-repo-url>
cd <the-folder-this-README-is-in>
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 1. Add your Groq key

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and set your key from <https://console.groq.com/keys>:

```ini
GROQ_API_KEY=gsk-your-key-here
GROQ_MODEL=openai/gpt-oss-20b
```

`.env` is gitignored. The agent will not start a conversation without a key, and says so
explicitly rather than answering from a template.

> The `gpt-oss` models only accept `temperature=1.0` on Groq and reject anything else with a 400.
> `GROQ_TEMPERATURE` defaults to `1.0` for that reason, and the adapter clamps and logs if you set
> something else. Switch to a `llama-*` model if you want to tune sampling.

### 2. Seed the database and build the vector index

```bash
python scripts/seed.py --reset
```

This creates `data/support.db` (10 customers, 10 subscriptions, 12 tickets), loads
`data/structured/*` into it, and indexes the 6 knowledge-base articles into Chroma (33 chunks).
The first run downloads a small ONNX MiniLM model for embeddings; if that fails it falls back to a
deterministic hashing embedder so the project still runs offline.

### 3. Run the API

```bash
uvicorn app.api.main:app --reload --port 8000
```

- API docs: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

### 4. Run the UI

In a second terminal:

```bash
streamlit run app/ui/app.py
```

It opens on <http://localhost:8501> and talks to the API at `UI_API_BASE` (default
`http://127.0.0.1:8000`).

---

## Try these

**Conflicting sources (the interesting one).** The KB says the refund window is 30 days;
`policies.json` says 14 for annual plans:

> What is the refund window for C-1001's annual plan?

The agent reports the conflict, states which source wins and why, and cites both.

**Irreversible action, human in the loop:**

> Issue a $75 goodwill credit to C-1001 for the June outage on T-5001 and add a note to the ticket.

The credit is held at `awaiting_approval`. Nothing is written to `account_credits` yet. Approve in
the UI (or `POST /chat/{session_id}/approve`) and the credit is written, the ticket event is
appended, and the reply tells you the credit id.

**Policy ceiling, enforced from data:**

> Refund C-1002 the full 900 USD they paid for last year's annual plan.

`900 > 250` (the ceiling in `policies.json`), so it asks for a supervisor and records the
approver's role. The approval panel in the UI offers a **role** dropdown built from
`approver_roles` in the policy file, so an over-ceiling refund can actually be approved from the
browser; the role travels with the decision into the action. Edit `partial_refund_ceiling_usd` in
the policy file and the behaviour changes with
no code change — there is a test that proves it.

**Read-only, nothing written:**

> How many seats does the Scale plan include?

---

## How it is put together

```
app/
  config.py            env-backed settings; every tunable is a Settings field
  schemas.py           every contract: AgentState pieces, tools, API payloads
  llm.py               GroqStructuredLLM (structured output + retries) and ScriptedLLM
  audit.py             append-only audit log, one row per node and per action
  memory.py            long-term memory in a LangGraph SqliteStore
  approval_store.py    durable mirror of pending/decided approval requests
  knowledge/
    chunking.py        heading-aware markdown splitter
    embeddings.py      ONNX MiniLM with a deterministic hashing fallback
    vector_store.py    persistent Chroma collection + semantic search
    sql_store.py       schema DDL, CSV seeding, read-only SQL guard
    policy_store.py    policies.json lookup, refund windows, approval rules
    catalog_store.py   catalog.csv lookup
  tools/
    read_tools.py      5 read-only retrieval tools
    action_registry.py ActionSpec, JSON-schema derivation, arg validation
    action_defs.py     10 actions, each doing a real write
  graph/
    state.py           AgentState, JSON-safe for checkpointing
    prompts.py         prompt templates; business rules are rendered from data
    nodes.py           all 13 nodes
    builder.py         graph assembly, conditional edges, SQLite checkpointer
    runner.py          run() and resume() entry points
  api/main.py          FastAPI service
  ui/app.py            Streamlit client
```

### Knowledge sources

| Source | What it holds | Authority |
| --- | --- | --- |
| `policies.json` | refund windows, SLA, escalation matrix, approval rules, credit ceilings | **highest** |
| `catalog.csv` | plans, prices, seats | high for pricing |
| `support.db` | live customers, subscriptions, tickets | **highest** for state |
| `kb/*.md` | prose articles and playbooks | explanatory; loses on numbers |

The model sees the sources ranked like that in its prompt, and every claim it makes carries
citations back to the row, file or chunk it came from.

### Tools

**Read (5):** `search_knowledge_base`, `query_operations_data`, `lookup_business_policy`,
`lookup_product_catalog`, `describe_operations_schema`.

**Act (10):**

| Action | Irreversible | Effect |
| --- | --- | --- |
| `create_ticket` | no | inserts a ticket |
| `add_ticket_note` | no | appends a `ticket_events` row |
| `update_ticket_status` | no | updates `tickets.status` + event |
| `escalate_ticket` | no | raises priority, writes an escalation notice |
| `send_customer_reply` | no | appends the reply to the outbox |
| `schedule_callback` | no | records a callback event |
| `export_case_report` | no | writes a case report file to `data/outbox/` |
| `apply_account_credit` | **yes** | inserts into `account_credits` |
| `cancel_subscription` | **yes** | cancels the subscription + writes a record |
| `issue_partial_refund` | **yes** | inserts into `refunds` |

The three irreversible ones are flagged in code *and* listed in `policies.json`; the policy list
wins, so making a new action approval-gated needs no code change. `issue_partial_refund` also
reads the ceiling and the allowed approver roles from the policy file at call time.

### Approval and audit

Irreversible work calls LangGraph's `interrupt()`. The process can be killed and restarted between
the pause and the decision; the pending request is still there, because state is checkpointed to
`data/checkpoints.db` and requests are mirrored in the `approvals` table.

Every node transition, retrieval, decision, approval and action outcome is written to `audit_log`
with the session id, actor, token counts and the payload. Read it back through
`GET /audit?session_id=...` or the UI's audit tab.

### Deliberate conflicts

`policies.json` says annual refunds are 14 days; `kb/billing-and-refunds.md` says 30 for all plans.
The catalog prices Growth at `$499`/month; the KB says `$450`. Both are intentional, and
`policies.json` documents them under `conflicts_on_purpose`. The agent is expected to surface the
disagreement and apply the higher-authority source, not to average them.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/chat` | run a turn; may return `awaiting_approval` |
| `POST` | `/chat/stream` | same turn as server-sent events (see below) |
| `POST` | `/chat/{session_id}/approve` | approve or reject, resumes the graph |
| `GET` | `/sources/documents` | markdown documents you can manage, with chunk counts |
| `POST` | `/sources/documents` | add or replace a document; reindexes |
| `DELETE` | `/sources/documents/{name}` | remove a document and purge its vectors |
| `GET` | `/approvals/pending` | outstanding requests across sessions |
| `GET` | `/approvals/recent` | decided requests |
| `GET` | `/audit` | audit rows, filterable by `session_id`/`event_type` |
| `GET` | `/stats` | row counts, index size, audit and approval totals |
| `GET` | `/sources` | health of all four sources |
| `GET` | `/actions` | the action catalogue with schemas and risk |
| `GET` | `/health` | liveness and whether a model is configured |
| `GET` | `/session/{session_id}/new` | clear a session's history |

### Streaming

`POST /chat/stream` runs the identical graph and returns `text/event-stream`. The
UI uses it so answers appear word by word; each frame is one `data:` line:

| Event | Payload | Meaning |
| --- | --- | --- |
| `start` | `session_id` | the turn has begun |
| `node` | `node` | a reasoning stage finished (intake, retrieve, …) |
| `token` | `text` | one answer delta |
| `done` | `response` | a full `ChatResponse`, identical to `/chat` |
| `error` | `error` | terminal failure; the status was already sent |

An approval pause arrives as a normal `done` with `status: awaiting_approval`, so a
client only needs the `token` and `done` branches. The response object is the same one
`/chat` returns, so a client can fall back to the blocking endpoint and get identical
results.

### Managing the knowledge base

The operations DB, policy file and catalog are fixed and have no write path. Only the
markdown articles in `data/knowledge/kb` are user-managed, through the UI sidebar
("Add or remove documents") or the API:

```bash
curl -X POST http://127.0.0.1:8000/sources/documents \
  -H "Content-Type: application/json" \
  -d '{"name": "escalation.md", "content": "# Escalation\n\nTier 2 owns Sev-1.", "overwrite": true}'

curl -X DELETE http://127.0.0.1:8000/sources/documents/escalation.md
```

Two behaviours worth knowing:

- **A mutation rebuilds the collection** rather than upserting. Chunk ids are derived
  from content, so a rewrite can never retire the ids its previous version used, and
  HNSW does not reliably reach vectors added to an already-built index. The knowledge
  base is a few dozen chunks, so a rebuild is cheap and buys exactness.
- **Names are validated, not sanitised.** Separators, dot segments, hidden names and
  Windows device names are rejected with `400`, and the resolved path is confirmed to
  sit directly inside the knowledge base directory.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many seats does the Scale plan include?"}'

curl -X POST http://127.0.0.1:8000/chat/<session_id>/approve \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "approver": "dana", "role": "billing-supervisor", "reason": "goodwill"}'
```

---

## Tests

```bash
python -m pytest              # 165 offline tests
python -m ruff check .
```

The offline suite runs against a temporary data directory with deterministic hashing embeddings, so
it never touches `data/` and never calls the network. It covers chunking, retrieval, the SQL
injection guard, every read tool, every action's side effects and validation, the policy-driven
thresholds, the full approval round trip, the state/response contracts, the Groq adapter
(temperature clamping, method fallback, token accounting, rate-limit fast-fail) and the HTTP
surface.

`tests/test_streaming.py` and `tests/test_api_stream_sources.py` cover the streamed and
document-management surfaces. Two of their assertions are worth calling out, because both
encode a bug that a weaker test would have passed:

- The concatenated `token` frames must equal the answer on the `done` frame, and must
  equal what the blocking `/chat` returns. The streamed responder joins deltas verbatim, so
  this is what forces the trailing-whitespace normalisation in the responder node.
- A removed or rewritten document must be *absent* from the collection, not merely
  outranked. `index_kb` is an upsert keyed on content-derived chunk ids, so a rewrite cannot
  retire its predecessor's ids on its own; `tests/test_api_stream_sources.py` checks the
  superseded text is gone from the indexed chunks directly.

`tests/test_ui.py` goes one step further and executes the Streamlit script through
`streamlit.testing.v1.AppTest` against a real API on a real port. This matters because
`streamlit run` answers HTTP 200 even when the script raises, so a health check cannot see a crash
on first paint — the sidebar once died on `TypeError: object of type 'int' has no len()` while the
server looked perfectly healthy. The same file pins a second crash of the same shape, a nested
`st.expander` raising `StreamlitAPIException` on every completed answer.

Ten further tests in `tests/test_live_groq.py` exercise the real model — structured output,
conflict reporting, and that an irreversible action really does halt for approval. They skip
themselves unless `GROQ_API_KEY` is set:

```bash
python -m pytest tests/test_live_groq.py -v
```

These are not cheap. Each one drives a whole graph, so the full file costs on the order of
150k+ tokens against `GROQ_API_KEY`'s daily allowance, and Groq enforces that per model per day
rather than per minute. When the budget is spent you get `Error code: 429 ... tokens per day`,
which the adapter raises as `LLMRateLimited` immediately instead of retrying across all three
structured-output methods — a spent budget cannot be recovered by trying harder. To verify the
provider after exhausting the quota, run a single cheap test:

```bash
python -m pytest tests/test_live_groq.py::test_model_is_reachable -v
```

---

## Configuration

All settings live in `.env` and are documented in `.env.example`. The ones worth knowing:

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | — | required for real reasoning |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | any model your key can reach |
| `GROQ_TEMPERATURE` | `1.0` | clamped to 1.0 for `gpt-oss`, which allows no other value |
| `AGENT_LLM_MODE` | `groq` | `fake` refuses model calls, for wiring checks |
| `EMBEDDING_BACKEND` | `auto` | `onnx` or `hashing` to force one |
| `RETRIEVAL_TOP_K` | `5` | chunks per source |

### Groq quota

Groq enforces its daily token limit **per model**, not per account, so a model that has hit its
cap is unblocked immediately by switching to another:

```
Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` ... (TPD): Limit 200000
```

A full run costs roughly 6,000–8,000 input tokens, and `MAX_ITERATIONS=4` means a verification
retry loop can push a single message towards 25,000. Against a 200k daily budget that is only a
handful of messages, so a live-test run will spend most of the day's allowance on its own.

When the budget is spent the adapter raises `LLMRateLimited` on the first response instead of
retrying across all three structured-output methods — a 429 is not something that retrying fixes.
| `MAX_ITERATIONS` | `4` | replan budget before the agent gives up |
| `MAX_PARALLEL_ACTIONS` | `5` | concurrent action ceiling |
| `AUTO_APPROVE_IRREVERSIBLE` | `0` | **leave at 0** unless you are testing |
| `UI_API_BASE` | `http://127.0.0.1:8000` | where the UI looks for the API |

---

## Notes and limitations

- `AUTO_APPROVE_IRREVERSIBLE=1` exists for demos. It is off by default and nothing should ship
  with it on.
- Refunds and credits move money, so the agent records an approver identity and role but does not
  talk to a payment processor.
- Memory is per-customer long-term recall, not a general user profile store.
- The knowledge base is six markdown files. Ingesting PDFs or a wiki export would mean a new loader
  in `app/knowledge/`; chunking, indexing and the graph would not change.

All code here is original. The seeded company, policies, tickets and knowledge-base articles are
fictional and were written for this project.
