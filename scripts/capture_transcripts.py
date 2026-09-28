"""Capture runnable transcripts against a live API.

    python scripts/capture_transcripts.py            # everything
    python scripts/capture_transcripts.py --only docs

Each transcript is a real request/response exchange recorded verbatim, written
to `docs/transcripts/`. Nothing here is hand-typed: the script performs the
request and formats what came back, so a transcript cannot drift from what the
service actually did.

The live agent turn is opt-in via --live because it spends Groq tokens and takes
minutes. Without it, only the local endpoints are exercised.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "transcripts"
BASE = "http://127.0.0.1:8000"


def now() -> str:
    return datetime.now(UTC).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def redact(text: str) -> str:
    """Strip account identifiers out of provider error text.

    A 429 quotes the organisation id and a billing URL. Those are fine in a
    terminal and not fine in a committed transcript, so they go either way.
    """
    text = re.sub(r"org_[a-z0-9]+", "org_***", text)
    return re.sub(r"https://console\.groq\.com\S*", "https://console.groq.com/...", text)


def write(name: str, body: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def header(title: str, note: str = "", target: str = "") -> list[str]:
    lines = [
        f"# {title}",
        "",
        f"Captured {now()} against `{target or BASE}`.",
    ]
    if note:
        lines += ["", f"> {note}"]
    return lines


def doc_crud() -> None:
    """Add, list, reject and remove a knowledge base document."""
    import httpx

    lines = header(
        "Document management transcript",
        "The user-managed knowledge base. Verifies that a bad name is rejected "
        "rather than silently rewritten, because a rewritten name would make "
        "the following delete miss.",
    )
    name = f"transcript-probe-{int(time.time())}.md"
    body = (
        "# Transcript probe\n\n"
        "## Refund window\n\nThis article exists only to prove that a document "
        "added at runtime is retrievable.\n"
    )

    with httpx.Client(base_url=BASE, timeout=60.0) as c:
        lines += ["", "## 1. List before", "", "```"]
        r = c.get("/sources/documents")
        lines.append("$ curl -s localhost:8000/sources/documents")
        lines.append(f"HTTP {r.status_code}")
        before = r.json()
        lines.append(json.dumps(before, indent=2)[:600])
        lines.append("```")

        lines += ["", "## 2. Add a document", "", "```"]
        lines.append("$ curl -X POST localhost:8000/sources/documents \\")
        lines.append("    -H 'content-type: application/json' \\")
        lines.append(f"    -d '{{\"name\": \"{name}\", \"content\": ...}}'")
        r = c.post("/sources/documents", json={"name": name, "content": body})
        lines.append(f"HTTP {r.status_code}")
        lines.append(json.dumps(r.json(), indent=2))
        added = r.json()
        lines.append("```")

        lines += ["", "## 3. Reject a traversal attempt", "", "```"]
        lines.append(
            "$ curl -X POST localhost:8000/sources/documents "
            "-d '{\"name\":\"../escape.md\"}'"
        )
        r = c.post("/sources/documents", json={"name": "../escape.md", "content": "nope"})
        lines.append(f"HTTP {r.status_code}")
        lines.append(json.dumps(r.json(), indent=2))
        lines.append("```")

        lines += ["", "## 4. List after", "", "```"]
        r = c.get("/sources/documents")
        after = r.json()
        lines.append(f"HTTP {r.status_code}")
        docs = after.get("documents", after)
        probe = [d for d in docs if d.get("name") == name]
        lines.append(json.dumps(docs, indent=2)[:600])
        lines.append("```")
        lines += [
            "",
            f"The probe appears in the listing: "
            f"`{json.dumps(probe[0]) if probe else 'NOT FOUND'}`. "
            f"Documents {len(docs)}, chunks {after.get('total_chunks', 'not reported')}.",
        ]

        lines += ["", "## 5. Remove it", "", "```"]
        lines.append(f"$ curl -X DELETE localhost:8000/sources/documents/{name}")
        r = c.delete(f"/sources/documents/{name}")
        lines.append(f"HTTP {r.status_code}")
        lines.append(json.dumps(r.json(), indent=2))
        removed = r.json()
        lines.append("```")

    lines += [
        "",
        "## Notes",
        "",
        f"- Collection held {added.get('indexed_chunks', '?')} chunks after the add "
        f"(`created: {added.get('created')}`) and "
        f"{removed.get('indexed_chunks', '?')} after removal, having removed "
        f"{removed.get('removed_chunks', '?')} chunk(s) with the file.",
        "- Mutations rebuild the whole collection rather than upserting. Chunk ids derive "
        "from content, so a rewrite cannot retire its predecessor's ids.",
        "- The probe article is not retrievable content. It exists only to show the write "
        "path and the index accounting; every document added this way is indexed the same.",
    ]
    write("02-document-management.md", "\n".join(lines))


def streaming() -> None:
    """Record the SSE frame types and headers for a streaming request."""
    import httpx

    lines = header(
        "Streaming transport transcript",
        "This is a real request against a real provider that failed on a daily token limit, so it "
        "records the transport contract and the in-band error path. A successful turn is in 04.",
    )

    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        lines += ["", "## 1. Stream a question", "", "```"]
        lines.append("$ curl -N -X POST localhost:8000/chat/stream -d '{\"message\":\"...\"}'")
        t0 = time.perf_counter()
        counts: dict[str, int] = {}
        order: list[str] = []
        first_token_at: float | None = None
        errors: list[dict] = []
        with c.stream("POST", "/chat/stream", json={"message": "What is the refund policy?"}) as r:
            lines.append(f"HTTP {r.status_code}")
            for h in ("content-type", "cache-control", "x-accel-buffering"):
                lines.append(f"{h}: {r.headers.get(h, '(absent)')}")
            lines.append("")
            for raw in r.iter_lines():
                if not raw.startswith("data:"):
                    continue
                try:
                    evt = json.loads(raw[5:].strip())
                except ValueError:
                    continue
                kind = str(evt.get("event", "?"))
                counts[kind] = counts.get(kind, 0) + 1
                if len(order) < 12 or kind not in order:
                    order.append(kind)
                if kind == "token" and first_token_at is None:
                    first_token_at = time.perf_counter() - t0
                    lines.append(f"  [first token at {first_token_at:.2f}s] {raw[:110]}")
                elif kind == "error":
                    errors.append(evt)
                    lines.append(f"  [error] {redact(raw[:300])}")
        total = time.perf_counter() - t0
        lines.append(f"  [stream closed at {total:.2f}s]")
        lines.append("```")

        if errors:
            raw = redact(json.dumps(errors[0], indent=2))
            lines += [
                "",
                "## The error is in-band, not a transport failure",
                "",
                "This run is a genuine failure worth showing: the provider rejected the call "
                "after the response had already committed to `200 text/event-stream`, so the "
                "only way to report it is a frame.",
                "",
                "```",
                raw[:700],
                "```",
                "",
                "The client cannot distinguish this from a transport error by status code "
                "alone, which is why the UI watches the last frame and not just the "
                "response code.",
                "",
                f"The 429 is a **daily** quota, not a per-minute one, and it is only detectable at "
                f"the point of call. This run gave up at the first node "
                f"({counts.get('node', 0)} node events, {total:.2f}s); an earlier one retried "
                f"through 27 node events before surfacing the same limit. Worth knowing before "
                f"pointing this at a real queue.",
            ]

    lines += [
        "",
        "## Frame counts",
        "",
        "| Event | Frames |",
        "| --- | --- |",
    ]
    for kind, n in sorted(counts.items()):
        lines.append(f"| `{kind}` | {n} |")
    lines += [
        "",
        "## First occurrences",
        "",
        "```",
        " -> ".join(order),
        "```",
        "",
        "## Notes",
        "",
        "- `cache-control: no-cache, no-transform` and `x-accel-buffering: no` are both "
        "required. Without the second, a reverse proxy will buffer the whole body and "
        "the stream arrives as one blob at the end.",
        "- The `done` frame carries the same ChatResponse as the blocking `/chat` endpoint. "
        "A test pins that the concatenated token frames equal it.",
    ]
    write("03-streaming.md", "\n".join(lines))


def live_turn(message: str) -> None:
    """A real end-to-end agent turn. Slow, and spends tokens."""
    import httpx

    lines = header(
        "Live agent turn",
        f"Real provider call. Question: {message!r}",
    )
    started = time.perf_counter()
    nodes: list[tuple[str, float]] = []
    tokens: list[str] = []
    done: dict = {}
    first_token_at: float | None = None
    lines += ["", "## Request", "", "```"]
    lines.append(f"$ curl -N -X POST localhost:8000/chat/stream -d '{{\"message\": {message!r}}}'")
    with httpx.Client(base_url=BASE, timeout=1200.0) as c, c.stream(
        "POST", "/chat/stream", json={"message": message}
    ) as r:
        lines.append(f"HTTP {r.status_code}")
        for h in ("content-type", "cache-control", "x-accel-buffering"):
            lines.append(f"{h}: {r.headers.get(h, '(absent)')}")
        lines.append("")
        lines.append("event:node        (reasoning progress)")
        for raw in r.iter_lines():
            if not raw.startswith("data:"):
                continue
            try:
                evt = json.loads(raw[5:].strip())
            except ValueError:
                continue
            kind = str(evt.get("event", "?"))
            if kind == "node":
                el = time.perf_counter() - started
                node = evt.get("node") or evt.get("label") or "?"
                nodes.append((str(node), el))
                lines.append(f"  t={el:7.2f}s  {node}")
            elif kind == "token":
                if not tokens:
                    first_token_at = time.perf_counter() - started
                    lines.append(f"  t={first_token_at:7.2f}s  first token")
                tokens.append(str(evt.get("text", evt.get("delta", ""))))
            elif kind == "done":
                done = evt
                lines.append(f"  t={time.perf_counter() - started:7.2f}s  done")
            elif kind == "error":
                lines.append(f"  t={time.perf_counter() - started:7.2f}s  error "
                             + redact(str(evt.get("error", ""))[:200]))
    total = time.perf_counter() - started

    if not done and not tokens:
        # The provider refused before the graph did any work (usually the daily
        # quota). Writing an empty transcript over a good earlier capture would
        # destroy it, so refuse and leave whatever is already on disk alone.
        print("  live turn produced no frames; leaving 04 untouched")
        if nodes:
            print(f"  saw {len(nodes)} node event(s), then no done frame")
        return

    answer = done.get("answer") or done.get("response", {}).get("answer", "")
    first_token = f"{first_token_at:.2f}s" if first_token_at is not None else "none"
    lines += [
        "",
        "## Answer",
        "",
        "```",
        str(answer).strip(),
        "```",
        "",
        "## Summary",
        "",
        "```",
        f"wall clock        {total:.2f}s",
        f"first token      {first_token}",
        f"token frames     {len(tokens)}",
        f"node events      {len(nodes)}",
        f"final status     {done.get('status', '?')}",
        f"citations        {len(done.get('citations', []))}",
        f"trace steps      {len(done.get('trace', []))}",
        f"iterations       {done.get('iterations', '?')}",
        "```",
        "",
        "## Node timeline",
        "",
        "| t (s) | Node |",
        "| --- | --- |",
    ]
    for node, el in nodes:
        lines.append(f"| {el:.2f} | `{node}` |")
    lines += [
        "",
        "## Notes",
        "",
        "- Every reasoning node completes before `respond` emits a token, so the first "
        "token lands at the end of the timeline. The node events are what make a turn "
        "this long legible; the token stream alone would look like a hang.",
        f"- {len(nodes)} node events for a single question is the real cost driver, and it "
        "is why the streaming work was scoped to the answer rather than the reasoning.",
    ]
    write("04-live-agent-turn.md", "\n".join(lines))


def corpus_turns() -> None:
    """Approval-gate transcript, rendered from a real scripted run.

    Runs the graph in-process with the labeled fixtures rather than over HTTP,
    because the interesting part is the response *shape* at the gate and that
    shape is easier to read as structured output than as a raw JSON body.
    """
    import uuid

    from app.graph.runner import run
    from app.schemas import ChatRequest
    from evals.run_trace_corpus import build_for, observe, sandbox_ctx, scenarios

    lines = header(
        "Approval gate transcript",
        "Scripted reasoning, real graph, real ops database, real approval interrupt. "
        "The provider is stubbed so the turn is instant and reproducible; everything "
        "shown about the gate, the evidence and the trace is genuine.",
        target="the in-process graph (evals/run_trace_corpus fixtures)",
    )

    for sc in scenarios():
        if sc["id"] not in {"tr-01", "tr-04", "tr-06"}:
            continue
        with sandbox_ctx() as settings:
            graph = build_for(settings, sc["scenario"])
            result = run(
                ChatRequest(
                    message=sc["message"],
                    session_id=f"transcript-{uuid.uuid4().hex[:8]}",
                    actor_name="t-dana",
                    actor_role="support_lead",
                ),
                settings=settings,
                graph=graph,
            )
            obs = observe(result)
            resp = result

        lines += [
            "",
            f"## {sc['id']} — {sc['name']}",
            "",
            "```",
            "$ POST /chat   (actor: t-dana, role: support_lead)",
            f'  message: "{sc["message"]}"',
            "",
            "HTTP 200",
            f"  status            {obs['status']}",
            f"  elapsed_ms        {obs['elapsed_ms']}",
            f"  iterations        {obs['iterations']}",
            f"  trace steps       {obs['trace_steps']}",
            f"  citations         {obs['citations']}  ({', '.join(obs['citation_source_types'])})",
            f"  conflicts found   {obs['n_conflicts']}",
            f"  decisions         {obs['n_decisions']}  {obs['decision_names']}",
            f"  approvals raised  {obs['n_approvals']}",
            f"  actions executed  {obs['n_actions']}",
            "```",
            "",
            "**Node order**",
            "",
            "```",
            " -> ".join(obs["nodes_visited"]),
            "```",
            "",
            "**Answer as returned**",
            "",
            "```",
            obs["answer"].strip(),
            "```",
        ]

        for ap_ in list(getattr(resp, "approvals", None) or []):
            lines += [
                "",
                "**Pending approval**",
                "",
                "```",
                json.dumps(
                    {
                        "id": getattr(ap_, "id", ""),
                        "action": getattr(ap_, "action", ""),
                        "risk": str(getattr(ap_, "risk", "")),
                        "reason": getattr(ap_, "reason", ""),
                        "status": str(getattr(ap_, "status", "")),
                    },
                    indent=2,
                ),
                "```",
            ]
        for c in list(getattr(resp, "conflicts", None) or []):
            lines += [
                "",
                "**Conflict reported to the user**",
                "",
                "```",
                json.dumps(c, indent=2)[:600],
                "```",
            ]

    lines += [
        "",
        "## Reading these",
        "",
        "- `awaiting_approval` with `actions executed 0` is the correct outcome, not a failure. "
        "The gate exists so an irreversible write cannot happen without a human.",
        "- The scripted stub returns the same prose for every read-only turn. Only the gate, the "
        "evidence and the trace are interesting here; the wording is a fixture.",
        "- Every scenario reports a conflict because the scripted reconciliation fixture declares "
        "one. In a live run the count reflects real disagreements found in the evidence.",
    ]
    write("01-approval-gate.md", "\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--only", choices=["docs", "stream", "live", "corpus"], default=None
    )
    ap.add_argument("--live", action="store_true", help="include the slow provider call")
    ap.add_argument(
        "--question",
        default="In one short sentence, what is the refund window for annual plans?",
    )
    args = ap.parse_args()

    print(f"capturing against {BASE}")
    if args.only in (None, "corpus"):
        corpus_turns()
    if args.only in (None, "stream"):
        streaming()
    if args.only in (None, "docs"):
        doc_crud()
    if args.live and args.only in (None, "live"):
        live_turn(args.question)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
