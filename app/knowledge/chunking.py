"""Markdown-aware chunking.

Splitting on structure rather than on a fixed width keeps each chunk attached to
the section heading it belongs to, so a retrieved chunk still means something
once the surrounding document is gone.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: list[str]
    chunk_index: int
    token_estimate: int

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_body(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body.strip()] if body.strip() else []

    sentences = [s for s in _SENTENCE_SPLIT.split(body) if s.strip()]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(sentence) > max_chars:
            for start in range(0, len(sentence), max_chars - overlap):
                parts.append(sentence[start : start + max_chars])
            current = ""
        else:
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()

    if current.strip():
        parts.append(current.strip())
    return [p for p in parts if p.strip()]


def chunk_markdown(
    text: str,
    *,
    source_name: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """Split a markdown document into heading-aware chunks."""
    heading_stack: list[str] = []
    current_heading: list[str] = []
    buffer: list[str] = []
    sections: list[tuple[list[str], str]] = []

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            if buffer:
                sections.append((list(current_heading), "\n".join(buffer)))
                buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [h for h in heading_stack]
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            current_heading = list(heading_stack)
        else:
            buffer.append(line)

    if buffer:
        sections.append((list(current_heading), "\n".join(buffer)))

    chunks: list[Chunk] = []
    index = 0
    for heading_path, body in sections:
        if not body.strip():
            continue
        for part in _split_body(body, chunk_size, chunk_overlap):
            header = f"{'#' * (len(heading_path))} " if heading_path else ""
            prefix = " ".join(f"{header}{h}" for h in heading_path)
            full = f"{prefix}\n\n{part}".strip()
            chunks.append(
                Chunk(
                    text=full,
                    heading_path=heading_path,
                    chunk_index=index,
                    token_estimate=_estimate_tokens(full),
                )
            )
            index += 1

    if not chunks and text.strip():
        chunks.append(
            Chunk(
                text=text.strip(),
                heading_path=[],
                chunk_index=0,
                token_estimate=_estimate_tokens(text),
            )
        )
    return chunks


def stable_chunk_id(source_name: str, chunk: Chunk) -> str:
    digest = hashlib.blake2b(
        f"{source_name}:{chunk.heading}:{chunk.chunk_index}:{chunk.text[:120]}".encode(),
        digest_size=8,
    ).hexdigest()
    return f"{digest}"
