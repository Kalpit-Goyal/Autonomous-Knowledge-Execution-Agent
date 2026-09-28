"""Embedding backends.

Primary backend is the ONNX MiniLM model that ships with Chroma: free, runs on
CPU, and after the first run the model is cached locally. When that model cannot
be loaded (no network on first run, no onnxruntime) we fall back to a
deterministic hashing embedder implemented here, so the whole agent stays
usable with zero downloads and zero cost.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_DIM = 384
_ASCII_FOLD = str.maketrans(
    {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "é": "e", "è": "e", "à": "a"}
)


def _normalise(text: str) -> str:
    return text.lower().translate(_ASCII_FOLD)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(_normalise(text))


def _bucket(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    index = value % _DIM
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return index, sign


class HashingEmbeddings(Embeddings):
    """Deterministic hashing embedder with no model download and no dependencies.

    Word and character trigram features are hashed into a fixed number of
    buckets with a signed contribution, weighted by sublinear term frequency and
    a length-normalised, then L2 normalised. It is not as semantically sharp as
    a trained model, but it is stable, cheap, and fully offline - which is what
    the fallback path needs.
    """

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _tokens(text)
        counts = Counter(tokens)
        for token, count in counts.items():
            index, sign = _bucket(token)
            vector[index] += sign * (1.0 + math.log(count))
        joined = " ".join(tokens)
        for i in range(len(joined) - 2):
            trigram = joined[i : i + 3]
            index, sign = _bucket(f"3:{trigram}")
            vector[index] += sign * 0.35
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _OnnxMiniLM(Embeddings):
    """Adapter around Chroma's bundled ONNX MiniLM embedding function.

    Chroma returns a numpy array rather than a list of lists, so the shape is
    normalised here instead of at every call site.
    """

    def __init__(self) -> None:
        from chromadb.utils import embedding_functions

        self._fn = embedding_functions.DefaultEmbeddingFunction()
        self.dim = 0

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        raw = self._fn(texts)
        array = np.asarray(raw, dtype="float64")
        if array.ndim == 1:
            matrix = array.reshape(1, -1)
        elif array.ndim == 2:
            matrix = array
        else:
            matrix = array.reshape(len(texts), -1)
        out = [[float(x) for x in row] for row in matrix]
        if out and not self.dim:
            self.dim = len(out[0])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


_lock = threading.Lock()
_cached: Embeddings | None = None


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return the configured embedding backend, memoised for the process."""
    global _cached
    with _lock:
        if _cached is not None:
            return _cached

        settings = settings or get_settings()
        backend = (settings.embedding_backend or "auto").strip().lower()

        if backend in {"auto", "onnx"}:
            try:
                _cached = _OnnxMiniLM()
                logger.info("Embedding backend: onnx-minilm")
                return _cached
            except Exception as exc:  # noqa: BLE001
                if backend == "onnx":
                    raise RuntimeError(
                        "EMBEDDING_BACKEND=onnx but the ONNX MiniLM model could not be loaded. "
                        "It downloads once (~80MB) then runs locally. Set "
                        "EMBEDDING_BACKEND=hashing to run with no download."
                    ) from exc
                logger.warning("ONNX MiniLM unavailable (%s); using hashing embeddings", exc)

        _cached = HashingEmbeddings()
        logger.info("Embedding backend: hashing")
        return _cached


def reset_embeddings_cache() -> None:
    global _cached
    with _lock:
        _cached = None
