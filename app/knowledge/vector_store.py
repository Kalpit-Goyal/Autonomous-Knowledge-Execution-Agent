"""Chroma-backed vector store over the markdown knowledge base.

Uses the embedded ``PersistentClient`` rather than a Chroma server: the
collection lives in ``data/chroma`` and is served in-process, so there is no
HTTP surface and nothing to run alongside the app.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.knowledge.chunking import chunk_markdown, stable_chunk_id
from app.knowledge.embeddings import get_embeddings
from app.schemas import Evidence, SourceType

logger = logging.getLogger(__name__)

COLLECTION = "meridian_kb"

_lock = threading.Lock()
_store: Chroma | None = None
_indexed_fingerprint: str | None = None


def get_vector_store(settings: Settings | None = None) -> Chroma:
    global _store
    with _lock:
        if _store is not None:
            return _store
        settings = settings or get_settings()
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embeddings(settings),
            persist_directory=str(settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        return _store


def reset_vector_store() -> None:
    """Drop the cached client.

    The client binds to one ``persist_directory`` on construction, so a test that
    repoints ``DATA_DIR`` would otherwise keep reading the previous index.

    chromadb also keeps a process-wide ``SharedSystemClient`` cache keyed by
    those settings, holding a strong reference (and an open sqlite handle) to
    every directory ever used. Without evicting it, a suite that gives each
    test its own data directory exhausts the process file-handle limit with
    ``OSError: [Errno 24] Too many open files``.
    """
    global _store, _indexed_fingerprint
    with _lock:
        _store = None
        _indexed_fingerprint = None
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001
        # Not every chromadb build exposes this; a leaked handle is survivable,
        # a failed import at teardown is not.
        logger.debug("could not clear the chromadb system cache", exc_info=True)


def _fingerprint(kb_dir: Path) -> str:
    import hashlib

    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(kb_dir.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def index_kb(
    *,
    force: bool = False,
    settings: Settings | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    """(Re)build the KB collection from the markdown files on disk."""
    global _indexed_fingerprint

    settings = settings or get_settings()
    kb_dir = settings.kb_dir
    if not kb_dir.exists():
        raise FileNotFoundError(f"knowledge base directory not found: {kb_dir}")

    fingerprint = _fingerprint(kb_dir)
    store = get_vector_store(settings)
    existing = store.get(limit=1)
    already_indexed = bool(existing.get("ids")) and _indexed_fingerprint == fingerprint

    if already_indexed and not force:
        return {
            "skipped": True,
            "reason": "index already matches the markdown sources",
            "documents": collection_stats(settings)["documents"],
            "sources": sorted({p.name for p in kb_dir.glob("*.md")}),
            "fingerprint": fingerprint,
        }

    documents: list[Document] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    sources: list[str] = []

    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_markdown(
            text,
            source_name=path.name,
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        ):
            doc_id = stable_chunk_id(path.name, chunk)
            documents.append(Document(page_content=chunk.text, metadata={}))
            metadatas.append(
                {
                    "source": path.name,
                    "source_type": SourceType.KB.value,
                    "heading": chunk.heading,
                    "chunk_index": chunk.chunk_index,
                    "token_estimate": chunk.token_estimate,
                    "doc_id": doc_id,
                }
            )
            ids.append(doc_id)
            sources.append(path.name)

    if documents:
        store.add_texts(
            texts=[d.page_content for d in documents],
            metadatas=metadatas,
            ids=ids,
            batch_size=64,
        )

    _indexed_fingerprint = fingerprint

    return {
        "skipped": False,
        "documents": len(documents),
        "sources": sorted(set(sources)),
        "fingerprint": fingerprint,
    }


def ensure_indexed(settings: Settings | None = None) -> dict[str, Any]:
    """Index on first use; a cheap no-op once the sources are unchanged.

    The fingerprint is process-local, so a restart re-runs the index pass. That
    is deliberate: it is an upsert keyed on stable chunk ids, so re-indexing
    after a restart refreshes the collection instead of duplicating it.
    """
    settings = settings or get_settings()
    return index_kb(force=False, settings=settings)


def search(
    query: str, top_k: int | None = None, settings: Settings | None = None
) -> list[Evidence]:
    settings = settings or get_settings()
    store = get_vector_store(settings)
    hits = store.similarity_search_with_score(query, k=top_k or settings.retrieval_top_k)

    evidence: list[Evidence] = []
    for doc, score in hits:
        meta = doc.metadata or {}
        distance = float(score)
        similarity = max(0.0, min(1.0, 1.0 - distance / 2.0)) if distance > 0 else 1.0
        source = meta.get("source", "unknown")
        heading = meta.get("heading") or ""
        evidence.append(
            Evidence(
                source_type=SourceType.KB,
                source_name=source,
                citation=f"kb/{source}" + (f" :: {heading}" if heading else ""),
                snippet=doc.page_content.strip(),
                score=round(similarity, 4),
                query=query,
                metadata={
                    "chunk_index": meta.get("chunk_index"),
                    "distance": round(distance, 6),
                    "doc_id": meta.get("doc_id"),
                },
            )
        )
    return evidence


def rebuild_kb(settings: Settings | None = None) -> dict[str, Any]:
    """Drop the collection and index every markdown file from scratch.

    Used after a user adds, removes or rewrites a document. Two reasons a
    plain upsert is not enough:

    * a rewrite changes the chunk ids, so the superseded chunks are never
      overwritten - only abandoned - and keep answering retrieval; and
    * HNSW does not reliably reach vectors added to an already-built index,
      so a freshly added document can be absent from search results even
      though ``get`` returns it.

    The KB is a few dozen chunks, so rebuilding is cheap and buys exactness.
    """
    settings = settings or get_settings()
    global _indexed_fingerprint
    store = get_vector_store(settings)
    _indexed_fingerprint = None

    # Empty the collection through the live client handle rather than deleting
    # the persist directory: on Windows the open handle keeps those files
    # locked, so an rmtree fails silently and the old vectors survive.
    try:
        existing = store.get(include=[]).get("ids") or []
    except Exception:  # noqa: BLE001
        existing = []
    if existing:
        store.delete(ids=list(existing))

    return index_kb(force=True, settings=settings)


def chunk_counts_by_source(settings: Settings | None = None) -> dict[str, int]:
    """Live chunk count per source name, read from the collection itself."""
    settings = settings or get_settings()
    store = get_vector_store(settings)
    counts: dict[str, int] = {}
    try:
        payload = store.get(include=["metadatas"])
        for meta in payload.get("metadatas") or []:
            name = (meta or {}).get("source")
            if name:
                counts[name] = counts.get(name, 0) + 1
    except Exception:  # noqa: BLE001
        logger.warning("could not read chunk metadata for the KB collection", exc_info=True)
        return {}
    return counts


def purge_source(name: str, settings: Settings | None = None) -> int:
    """Delete every chunk belonging to ``name`` and return how many went.

    ``index_kb`` is an upsert keyed on stable chunk ids, so deleting a markdown
    file on its own would leave its vectors behind and keep answering retrieval.
    """
    settings = settings or get_settings()
    store = get_vector_store(settings)
    try:
        payload = store.get(where={"source": name}, include=[])
        ids = list(payload.get("ids") or [])
    except Exception:  # noqa: BLE001
        logger.warning("could not look up chunks for %s", name, exc_info=True)
        return 0
    if not ids:
        return 0
    store.delete(ids=ids)
    return len(ids)


def collection_stats(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    store = get_vector_store(settings)
    try:
        count = store._collection.count()
    except Exception:  # noqa: BLE001
        count = len(store.get(include=[]).get("ids", []))
    return {
        "collection": COLLECTION,
        "documents": int(count),
        "backend": settings.embedding_backend,
    }
