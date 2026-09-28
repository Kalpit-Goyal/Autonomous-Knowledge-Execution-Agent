"""Streamlit front end.

The UI is a thin HTTP client. It never touches the graph or the database
directly, which keeps the two processes independently runnable and means the
UI can be restarted without disturbing an approval that is already waiting.

Three things are on screen at once, because the point of the agent is
inspectability:

* the answer, with its citations;
* the reasoning trace, step by step, including what it decided *not* to do;
* the approval queue, when something irreversible is pending.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import httpx
import streamlit as st

st.set_page_config(
    page_title="Meridian Cloud Support Agent",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("UI_API_BASE", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("UI_API_TIMEOUT", "300"))


# --------------------------------------------------------------------------- #
# api client
# --------------------------------------------------------------------------- #


@st.cache_resource(ttl=30)
def client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=TIMEOUT)


def api(method: str, path: str, **kwargs) -> Any:
    try:
        response = client().request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        st.error(
            f"Could not reach the agent API at {API_BASE}.\n\n"
            f"Start it with `python -m uvicorn app.api.main:app --port 8000`.\n\n"
            f"Details: {exc}"
        )
        return None
    if response.status_code >= 400:
        detail = response.text
        with contextlib.suppress(ValueError):
            detail = response.json().get("detail", detail)
        st.error(f"{method} {path} failed ({response.status_code}): {detail}")
        return None
    return response.json()


def stream_chat(
    message: str,
    session_id: str | None,
    placeholder: Any,
    progress: Any,
) -> dict[str, Any] | None:
    """Stream one turn, painting answer deltas into `placeholder` as they arrive.

    Returns the finished ChatResponse, or None if the stream failed. The server
    sends the same response object on `done` that `/chat` would have returned, so
    the caller can render normally afterwards.
    """
    payload = {"message": message, "session_id": session_id or None}
    buffer: list[str] = []
    final: dict[str, Any] | None = None

    try:
        with client().stream("POST", "/chat/stream", json=payload, timeout=TIMEOUT) as response:
            if response.status_code >= 400:
                response.read()
                with contextlib.suppress(ValueError):
                    detail = response.json().get("detail", response.text)
                st.error(f"POST /chat/stream failed ({response.status_code}): {detail}")
                return None

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                kind = event.get("event")

                if kind == "token":
                    buffer.append(event.get("text", ""))
                    # Re-rendering the whole buffer each time is what makes the
                    # answer appear to type itself out.
                    placeholder.markdown("".join(buffer).strip() or "…")
                elif kind == "node":
                    progress.caption(f"Working: {event.get('node')}")
                elif kind == "start":
                    progress.caption("Working…")
                elif kind == "done":
                    final = event.get("response")
                elif kind == "error":
                    st.error(f"The agent failed: {event.get('error', 'unknown error')}")
                    return None
    except httpx.HTTPError as exc:
        st.error(
            f"Could not reach the agent API at {API_BASE}.\n\n"
            f"Start it with `python -m uvicorn app.api.main:app --port 8000`.\n\n"
            f"Details: {exc}"
        )
        return None

    progress.empty()
    if final is None:
        st.error("The stream ended before the agent sent a result.")
        return None
    return final


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #

if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

RISK_COLOURS = {"low": "green", "medium": "orange", "high": "red"}
STATUS_LABELS = {
    "completed": "Completed",
    "awaiting_approval": "Waiting for human approval",
    "needs_input": "Needs more information",
    "error": "Error",
}


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### Agent status")
    health = api("GET", "/health")
    if health:
        if health.get("llm_configured"):
            st.success(f"Groq ready ({health['llm_model']})")
        else:
            st.error("No GROQ_API_KEY configured")
            st.caption(health.get("llm_hint", ""))
        st.caption(f"Embeddings: {health.get('embedding_backend')}")
        st.caption(f"Max iterations: {health.get('max_iterations')}")
        if health.get("auto_approve_irreversible"):
            st.warning("Irreversible actions are auto-approved in this configuration.")

    st.divider()
    st.markdown("### Knowledge sources")
    sources = api("GET", "/sources")
    if sources:
        kb = sources["knowledge_base"]["indexed"]
        st.markdown(
            f"- **Knowledge base** — {kb['documents']} passages (`{kb['backend']}`)"
        )
        st.markdown("- **Operations DB** — customers, subscriptions, tickets")
        st.markdown(f"- **Policy** — {sources['policy'].get('version', 'n/a')} (authoritative)")
        # `rows` is a count, not a list of row objects.
        row_count = sources["catalog"].get("rows", 0)
        st.markdown(f"- **Catalog** — {row_count} plan rows")
    else:
        # Otherwise the heading sits above nothing and looks like a bug rather
        # than an unreachable API.
        st.caption(f"Could not reach the agent API at {API_BASE}.")

    st.divider()
    st.markdown("### Manage knowledge base")
    with st.expander("Add or remove documents", expanded=False):
        st.caption(
            "Documents here are markdown files the agent can retrieve. "
            "The operations DB, policy and catalog are fixed and not editable here."
        )
        listing = api("GET", "/sources/documents")
        if listing:
            for doc in listing.get("documents", []):
                label = f"`{doc['name']}` — {doc['chunks']} passages, {doc['bytes'] // 1024} KB"
                # Delete is per-row, so each button needs its own stable key.
                if st.button(f"Remove {doc['name']}", key=f"rm_{doc['name']}"):
                    result = api("DELETE", f"/sources/documents/{doc['name']}")
                    if result is not None:
                        st.success(
                            f"Removed `{result['name']}` "
                            f"({result.get('removed_chunks', 0)} passages purged)."
                        )
                        st.rerun()
                st.caption(label)
            if not listing.get("documents"):
                st.caption("No documents yet.")

        uploaded = st.file_uploader("Upload a .md file", type=["md"], key="kb_upload")
        pasted_name = st.text_input("Name", value="", placeholder="my-doc.md", key="kb_name")
        pasted_body = st.text_area("Markdown", value="", height=140, key="kb_body")
        if st.button("Add document", key="kb_add"):
            chosen = uploaded if uploaded is not None else None
            name = pasted_name.strip() or (chosen.name if chosen is not None else "")
            if chosen is not None:
                body = chosen.getvalue().decode("utf-8", errors="replace")
            else:
                body = pasted_body
            if not name:
                st.error("Give the document a name, or upload a file.")
            elif not body.strip():
                st.error("The document is empty.")
            else:
                result = api(
                    "POST",
                    "/sources/documents",
                    json={"name": name, "content": body, "overwrite": True},
                )
                if result is not None:
                    st.success(
                        f"Saved `{result['name']}` "
                        f"({result.get('indexed_chunks')} passages)."
                    )
                    st.rerun()

    st.divider()
    st.markdown("### Awaiting approval")
    pending = api("GET", "/approvals/pending?limit=20") or []
    if not pending:
        st.caption("Nothing waiting.")
    for item in pending:
        st.warning(f"`{item['action']}` — {item.get('session_id', '')}")

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = ""
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id or 'not started'}`")
    st.caption(f"API: {API_BASE}")


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for ev in citations:
            st.markdown(
                f"**{ev['citation']}**  ·  `{ev['source_type']}`  ·  score {ev['score']:.2f}"
            )
            body = ev["snippet"]
            st.caption(body if len(body) <= 400 else body[:400] + " …")


def render_trace(response: dict[str, Any]) -> None:
    steps = response.get("trace") or []
    if not steps:
        return
    with st.expander(f"Reasoning trace ({len(steps)} steps)", expanded=False):
        for index, step in enumerate(steps, start=1):
            st.markdown(f"**{index}. {step['node']}** — {step['label']}")
            if step.get("detail"):
                st.write(step["detail"])
            data = step.get("data") or {}
            for key, value in data.items():
                if value in (None, [], {}, ""):
                    continue
                if isinstance(value, (dict, list)):
                    # No st.expander here: expanders cannot nest, and this
                    # block already lives inside the trace expander. st.json
                    # is its own collapsible viewer, so it is safe to embed.
                    st.caption(key)
                    st.json(value, expanded=False)
                else:
                    st.caption(f"{key}: {value}")
            st.divider()


def render_conflicts(response: dict[str, Any]) -> None:
    conflicts = response.get("conflicts") or []
    if not conflicts:
        return
    with st.expander(f"Conflicting sources ({len(conflicts)})", expanded=True):
        for item in conflicts:
            summary = item.get("summary") if isinstance(item, dict) else str(item)
            st.markdown(f"- {summary}")


def render_actions(response: dict[str, Any]) -> None:
    outcomes = response.get("actions") or []
    if not outcomes:
        return
    st.markdown("#### Actions taken")
    for outcome in outcomes:
        colour = "green" if outcome["ok"] else "red"
        if outcome.get("executed_by") == "skipped":
            colour = "grey"
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"**`{outcome['action']}`**")
            right.markdown(
                f":{colour}[{outcome.get('executed_by', 'agent')}] "
                f"{'ok' if outcome['ok'] else outcome.get('error', 'failed')}"
            )
            if outcome.get("why"):
                st.caption(outcome["why"])
            if outcome.get("result"):
                st.json(outcome["result"], expanded=False)
            if outcome.get("approval_id"):
                st.caption(f"approval `{outcome['approval_id']}`")


def render_decisions(response: dict[str, Any]) -> None:
    decisions = response.get("decisions") or []
    if not decisions:
        return
    with st.expander(f"Decisions ({len(decisions)})", expanded=False):
        for decision in decisions:
            colour = RISK_COLOURS.get(decision.get("risk", "low"), "grey")
            st.markdown(f"**`{decision['action']}`** :{colour}[{decision['risk']}]")
            st.write(decision.get("why", ""))
            if decision.get("evidence_refs"):
                st.caption("evidence: " + ", ".join(decision["evidence_refs"]))
            if decision.get("depends_on"):
                st.caption("depends on: " + ", ".join(decision["depends_on"]))


def _approver_roles() -> list[str]:
    """Roles the policy file accepts, so the dropdown cannot offer a wrong one."""
    sources = api("GET", "/sources") or {}
    roles = (sources.get("policy") or {}).get("approver_roles") or []
    return [str(r) for r in roles]


def render_approval_panel(response: dict[str, Any]) -> None:
    pending = [a for a in (response.get("approvals") or []) if a.get("status") == "pending"]
    if not pending:
        return
    st.warning(f"{len(pending)} action(s) are waiting for a human decision.")
    session_id = response.get("session_id")
    roles = _approver_roles()
    for index, approval in enumerate(pending):
        with st.container(border=True):
            st.markdown(f"**`{approval['action']}`**  risk: `{approval.get('risk')}`")
            st.caption(approval.get("reason_for_approval", ""))
            if approval.get("why"):
                st.write(f"Why the agent wants this: {approval['why']}")
            if approval.get("args"):
                st.json(approval["args"], expanded=False)
            st.caption(f"approval id `{approval['id']}`")

            approver = st.text_input(
                "Approver", value="dana@meridian.example", key=f"approver_{approval['id']}"
            )
            # Some actions (a refund above the policy ceiling) additionally need
            # the approver's role, so offer exactly the roles policy allows.
            role = ""
            if roles:
                role = st.selectbox(
                    "Approver role",
                    ["(not set)", *roles],
                    key=f"role_{approval['id']}",
                    help="Required when policy demands a specific role, e.g. a "
                    "refund above the ceiling.",
                )
            reason = st.text_input("Reason", key=f"reason_{approval['id']}")
            col_a, col_r = st.columns(2)
            if col_a.button("Approve", key=f"yes_{approval['id']}_{index}", type="primary"):
                _submit(session_id, approval, True, approver, reason, role)
            if col_r.button("Reject", key=f"no_{approval['id']}_{index}"):
                _submit(session_id, approval, False, approver, reason, role)


def _submit(
    session_id: str,
    approval: dict[str, Any],
    approved: bool,
    approver: str,
    reason: str,
    role: str = "",
) -> None:
    st.session_state.last_response = api(
        "POST",
        f"/chat/{session_id}/approve",
        json={
            "decisions": [
                {
                    "action": approval["action"],
                    "approved": approved,
                    "approver": approver,
                    "role": "" if role.startswith("(not set)") else role,
                    "reason": reason,
                }
            ]
        },
    )
    st.rerun()


def render_response(response: dict[str, Any]) -> None:
    status = response.get("status", "completed")
    label = STATUS_LABELS.get(status, status)
    icon = {"completed": "✅", "awaiting_approval": "⏸️", "needs_input": "❓", "error": "❌"}.get(
        status, "•"
    )
    st.markdown(f"{icon} **{label}** · {response.get('elapsed_ms', 0)} ms")

    if response.get("error"):
        st.error(response["error"])
    if response.get("answer"):
        st.markdown(response["answer"])
    if status == "needs_input" and response.get("user_question"):
        st.info(response["user_question"])

    render_conflicts(response)
    render_approval_panel(response)
    render_actions(response)
    render_decisions(response)
    render_citations(response.get("citations") or [])
    render_trace(response)

    verification = response.get("verification")
    if verification:
        with st.expander("Verification", expanded=False):
            st.write(verification.get("reasoning", ""))
            st.caption(
                f"achieved: {verification.get('achieved')} · "
                f"confidence {verification.get('confidence')}"
            )
            if verification.get("missing"):
                st.write("missing: " + ", ".join(verification["missing"]))


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #

st.title("Meridian Cloud Support Agent")
st.caption(
    "Describe a situation or ask a question. The agent plans its own retrieval, "
    "reconciles conflicting sources, and asks a human before anything irreversible."
)

health = api("GET", "/health")
if health and not health.get("llm_configured"):
    st.error(
        "The API has no `GROQ_API_KEY`, so it will refuse requests. "
        "Copy `.env.example` to `.env`, add your Groq key, and restart the API."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

example = st.chat_input(
    "Ask a question or describe what needs handling",
    key="chat_input",
)

if example:
    st.session_state.messages.append({"role": "user", "content": example})
    with st.chat_message("user"):
        st.markdown(example)

    with st.chat_message("assistant"):
        # The placeholder is painted in place while tokens arrive, so the answer
        # grows word by word instead of appearing after a blocking wait.
        streamed = st.empty()
        progress = st.empty()
        response = stream_chat(
            example,
            st.session_state.session_id or None,
            streamed,
            progress,
        )
        streamed.empty()

    if response:
        st.session_state.session_id = response.get("session_id", "")
        st.session_state.last_response = response
        if response.get("answer"):
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )
        render_response(response)

if st.session_state.last_response and not example:
    with st.chat_message("assistant"):
        render_response(st.session_state.last_response)
