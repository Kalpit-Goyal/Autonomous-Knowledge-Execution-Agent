"""FastAPI service.

The API is a thin shell over the graph. Its only real jobs are to translate
HTTP into graph invocations, to surface the approval queue, and to expose the
diagnostics that make the agent's behaviour inspectable without reading logs.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import Field

from app import approval_store
from app.audit import query as audit_query
from app.audit import stats as audit_stats
from app.config import get_settings
from app.graph.runner import new_session_id, resume, run, stream_run
from app.knowledge import sql_store
from app.knowledge.catalog_store import catalog_summary
from app.knowledge.policy_store import policy_summary
from app.knowledge.sources import SourceError, list_documents, remove_document
from app.knowledge.sources import add_document as add_kb_document
from app.knowledge.vector_store import collection_stats, ensure_indexed
from app.llm import LLMUnavailable
from app.schemas import (
    ApproveRequest,
    ChatRequest,
    ChatResponse,
    SourceCreate,
    SourceList,
    SourceMutation,
)
from app.tools import ACTIONS, needs_approval, registry_as_prompt_block

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Index the knowledge base at boot so the first request is not the one that
    # pays for embedding the corpus.
    try:
        ensure_indexed()
    except Exception as exc:  # noqa: BLE001
        logger.warning("startup indexing failed: %s", exc)
    yield


app = FastAPI(
    title="Meridian Cloud Support Agent",
    description=(
        "An autonomous knowledge-execution agent. It plans, retrieves from several "
        "internal sources, reconciles conflicts, decides and executes actions, and "
        "pauses for human approval on anything irreversible."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeRequest(ApproveRequest):
    decisions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-action decisions. Each item: {action, approved, approver, reason}. "
            "Omit to apply `approved` to everything outstanding."
        ),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    llm_ready = settings.has_llm_key
    return {
        "status": "ok",
        "llm_configured": llm_ready,
        "llm_provider": "groq",
        "llm_model": settings.groq_model,
        "llm_hint": (
            "Groq key found."
            if llm_ready
            else "No GROQ_API_KEY. The API starts but reasoning calls will fail. "
            "Copy .env.example to .env and add your key."
        ),
        "embedding_backend": settings.embedding_backend,
        "max_iterations": settings.max_iterations,
        "auto_approve_irreversible": settings.auto_approve_irreversible,
    }


def _require_llm() -> None:
    if not settings.has_llm_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured. Copy .env.example to .env, add your key, "
                "and restart the API."
            ),
        )


def _require_message(request: ChatRequest) -> None:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")


def _sse(event: dict[str, Any]) -> str:
    """Frame one event for text/event-stream.

    A JSON payload is embedded as a `data:` line, so the whole frame is a
    single line and no payload can split the stream.
    """
    return f"event: {event['event']}\ndata: {json.dumps(event)}\n\n"


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream one turn as server-sent events.

    Events are `start`, `node` (reasoning progress), `token` (answer deltas)
    and exactly one terminal `done` or `error`. The `done` payload is the same
    ChatResponse the blocking `/chat` returns, so a client can fall back to
    that endpoint and get identical results.
    """
    _require_message(request)
    _require_llm()

    def _generate() -> Any:
        try:
            for event in stream_run(request, settings=settings):
                yield _sse(event)
        except LLMUnavailable as exc:
            # The status has already been sent, so the failure has to arrive as
            # an in-band error event rather than an HTTP status code.
            yield _sse({"event": "error", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("streamed run failed")
            yield _sse({"event": "error", "error": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Stops nginx and similar proxies from buffering the stream into a
            # single chunk, which would defeat the point of streaming.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sources/documents", response_model=SourceList)
def source_documents() -> dict[str, Any]:
    """Markdown documents a user may manage, with live chunk counts."""
    return list_documents(settings=settings)


@app.post("/sources/documents", response_model=SourceMutation, status_code=201)
def create_source_document(payload: SourceCreate) -> dict[str, Any]:
    try:
        return add_kb_document(
            payload.name,
            payload.content,
            overwrite=payload.overwrite,
            settings=settings,
        )
    except SourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/sources/documents/{name}", response_model=SourceMutation)
def delete_source_document(name: str) -> dict[str, Any]:
    try:
        return remove_document(name, settings=settings)
    except SourceError as exc:
        # An unknown document is a 404, not a 400: the request was well formed.
        status = 404 if "not in the knowledge base" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    _require_message(request)
    _require_llm()
    try:
        return run(request, settings=settings)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/chat/{session_id}/approve", response_model=ChatResponse)
def approve(session_id: str, request: ResumeRequest) -> ChatResponse:
    if not settings.has_llm_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured.")

    outstanding = [
        a for a in approval_store.pending(settings=settings) if a["session_id"] == session_id
    ]
    if not outstanding:
        raise HTTPException(
            status_code=404, detail=f"no pending approval for session {session_id}"
        )

    if request.decisions:
        decisions = request.decisions
    else:
        decisions = [
            {
                "action": item["action"],
                "approved": request.approved,
                "approver": request.approver,
                "reason": request.reason,
            }
            for item in outstanding
        ]
    for item in decisions:
        item.setdefault("approver", request.approver)
        item.setdefault("role", request.role)
        item.setdefault("reason", request.reason)
        item.setdefault("approved", request.approved)

    try:
        return resume(session_id, decisions, settings=settings)
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/approvals/pending")
def pending_approvals(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    return approval_store.pending(settings=settings, limit=limit)


@app.get("/approvals/recent")
def recent_approvals(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    return approval_store.recent(settings=settings, limit=limit)


@app.get("/audit")
def audit(
    session_id: str | None = None,
    event_type: str | None = None,
    action: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    return audit_query(
        session_id=session_id,
        event_type=event_type,
        action=action,
        limit=limit,
        settings=settings,
    )


@app.get("/stats")
def stats() -> dict[str, Any]:
    pending = approval_store.pending(settings=settings, limit=500)
    return {
        "audit": audit_stats(settings=settings),
        "approvals": {
            "pending": len(pending),
            "waiting_on": sorted({a["action"] for a in pending}),
        },
        "knowledge": collection_stats(settings=settings),
        "policy": policy_summary(),
        "catalog": catalog_summary(),
        "rows": sql_store.row_counts(settings=settings),
    }


@app.get("/sources")
def sources() -> dict[str, Any]:
    """Everything the agent can read, and where it comes from."""
    return {
        "knowledge_base": {
            "path": str(settings.kb_dir),
            "indexed": collection_stats(settings=settings),
        },
        "operations_db": {
            "path": str(settings.db_path),
            "schema": sql_store.schema_description(settings=settings),
        },
        "policy": {"path": str(settings.policies_path), **policy_summary()},
        "catalog": {"path": str(settings.catalog_path), **catalog_summary()},
    }


@app.get("/actions")
def actions() -> dict[str, Any]:
    return {
        "catalogue": registry_as_prompt_block(),
        "actions": [
            {
                "name": spec.name,
                "description": spec.description,
                "effect": spec.effect,
                "risk": spec.risk.value,
                "irreversible": spec.irreversible,
                "requires_approval": needs_approval(spec.name),
                "parameters": spec.args_model.model_json_schema(),
            }
            for spec in sorted(ACTIONS.values(), key=lambda s: s.name)
        ],
    }


@app.get("/session/{session_id}/new")
def new_session() -> dict[str, str]:
    return {"session_id": new_session_id()}
