"""User-managed knowledge-base documents.

The fixed sources (ops DB, policy, catalog) are part of the application's
contract and deliberately have no write path. This module manages only the
markdown files a user may add or remove under ``data/knowledge/kb``.

Two invariants matter more than convenience here:

* a caller-supplied name never escapes the KB directory, and
* removing a file also removes its vectors, because ``index_kb`` only upserts.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.knowledge.chunking import chunk_markdown
from app.knowledge.vector_store import chunk_counts_by_source, rebuild_kb

MAX_DOCUMENT_BYTES = 512 * 1024

# Deliberately strict. A document name becomes both a filename and a Chroma
# filter value, so separators, dot segments and reserved characters are refused
# rather than sanitised: silently rewriting a name would make a delete miss.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$")
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class SourceError(ValueError):
    """A rejected add/remove request. Carries an HTTP-friendly reason."""


def _kb_dir(settings: Settings) -> Path:
    return settings.kb_dir


def validate_name(raw: str) -> str:
    """Return a safe ``*.md`` filename or raise :class:`SourceError`."""
    name = (raw or "").strip()
    if not name:
        raise SourceError("a document name is required")
    if name != raw.strip() or "/" in name or "\\" in name or "\x00" in name:
        raise SourceError("a document name must not contain path separators")
    if name in {".", ".."} or name.startswith("."):
        raise SourceError("a document name must not start with a dot")
    if not _SAFE_NAME.match(name):
        raise SourceError(
            "a document name may only contain letters, digits, spaces, dots, "
            "dashes and underscores"
        )
    stem = name[:-3] if name.lower().endswith(".md") else name
    if stem.upper() in _RESERVED:
        raise SourceError(f"{stem} is a reserved device name")
    if not stem.lower().endswith(".md"):
        # Append to the stem, not to the whole name, or "notes.md" would become
        # "notes.md.md".
        name = f"{stem}.md"
    return name


def _resolve(settings: Settings, name: str) -> Path:
    """Resolve ``name`` inside the KB dir, refusing anything that escapes it.

    The regex already blocks separators, but this is the check that actually
    guarantees containment, so it stays even if the pattern is ever relaxed.
    """
    kb_dir = _kb_dir(settings).resolve()
    target = (kb_dir / name).resolve()
    if target.parent != kb_dir:
        raise SourceError("a document must live directly in the knowledge base")
    return target


def list_documents(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    kb_dir = _kb_dir(settings)
    if not kb_dir.exists():
        return {"documents": [], "count": 0}
    counts = chunk_counts_by_source(settings)
    rows = []
    for path in sorted(kb_dir.glob("*.md")):
        stat = path.stat()
        rows.append(
            {
                "name": path.name,
                "bytes": int(stat.st_size),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                "chunks": int(counts.get(path.name, 0)),
            }
        )
    return {"documents": rows, "count": len(rows), "directory": str(kb_dir)}


def read_document(name: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    target = _resolve(settings, validate_name(name))
    if not target.is_file():
        raise SourceError(f"{target.name} is not in the knowledge base")
    return target.read_text(encoding="utf-8")


def add_document(
    name: str, content: str, *, overwrite: bool = False, settings: Settings | None = None
) -> dict[str, Any]:
    settings = settings or get_settings()
    safe = validate_name(name)
    text = (content or "").strip()
    if not text:
        raise SourceError("a document must not be empty")
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise SourceError(
            f"a document may be at most {MAX_DOCUMENT_BYTES // 1024} KB "
            f"(this one is {len(encoded) // 1024} KB)"
        )

    target = _resolve(settings, safe)
    created = not target.exists()
    if created is False and not overwrite:
        raise SourceError(f"{safe} already exists; pass overwrite to replace it")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    index = rebuild_kb(settings=settings)

    return {
        "name": safe,
        "created": created,
        "bytes": int(target.stat().st_size),
        "indexed_chunks": index.get("documents", 0),
        "sources": index.get("sources", []),
    }


def remove_document(name: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    safe = validate_name(name)
    target = _resolve(settings, safe)
    if not target.is_file():
        raise SourceError(f"{safe} is not in the knowledge base")

    before = len(
        chunk_markdown(
            target.read_text(encoding="utf-8"),
            source_name=safe,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )
    target.unlink()
    # The vectors outlive the file: index_kb only upserts on stable chunk ids,
    # so removing the markdown is not enough on its own.
    index = rebuild_kb(settings=settings)

    return {
        "name": safe,
        "removed_chunks": before,
        "indexed_chunks": index.get("documents", 0),
    }
