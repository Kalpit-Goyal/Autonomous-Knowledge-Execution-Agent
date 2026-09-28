"""Long-term memory.

LangGraph ships an in-memory store but no persistent adapter, so this module
implements :class:`SqliteLongTermStore` against
:class:`langgraph.store.base.BaseStore` directly. It satisfies the store contract
(``batch``/``abatch`` plus the sync primitives) and adds two things the stock
store does not do: it persists across restarts, and ``search`` can rank by
embedding similarity instead of returning rows in insertion order.

Namespaces used by the agent:

* ``("agent", "long_term", "facts")``  - cross-session facts about the domain
* ``("agent", "long_term", "user", user_id)`` - per-user profile and preferences
* ``("agent", "long_term", "session", session_id)`` - what happened in a session
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

from app.config import Settings, get_settings
from app.knowledge.embeddings import get_embeddings

logger = logging.getLogger(__name__)

FACTS_NS: tuple[str, ...] = ("agent", "long_term", "facts")
SESSION_NS: tuple[str, ...] = ("agent", "long_term", "session")
USER_NS: tuple[str, ...] = ("agent", "long_term", "user")

DDL = """
CREATE TABLE IF NOT EXISTS store_items (
    namespace   TEXT NOT NULL,
    key         TEXT NOT NULL,
    value_json  TEXT NOT NULL,
    text        TEXT,
    embedding   BLOB,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    expires_at  TEXT,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_store_ns ON store_items (namespace);
"""


def _ns_str(namespace: tuple[str, ...]) -> str:
    return json.dumps(list(namespace), separators=(",", ":"))


def _ns_tuple(raw: str) -> tuple[str, ...]:
    return tuple(json.loads(raw))


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(raw: str | None) -> datetime:
    if not raw:
        return _now()
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return _now()


def _flatten(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in value.items():
        if isinstance(v, (dict, list)):
            parts.append(f"{k}: {json.dumps(v, default=str)}")
        else:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


class SqliteLongTermStore(BaseStore):
    """Persistent, embedding-ranked long-term memory for the agent."""

    def __init__(
        self,
        db_path: str | None = None,
        *,
        settings: Settings | None = None,
        index: bool = True,
    ) -> None:
        settings = settings or get_settings()
        self._path = db_path or str(settings.data_dir / "memory.db")
        self._lock = threading.RLock()
        self._index_enabled = index
        self._embeddings = get_embeddings(settings) if index else None
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(self._path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(DDL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _embed(self, text: str) -> bytes | None:
        if not self._embeddings or not text.strip():
            return None
        try:
            vector = self._embeddings.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory embedding failed, storing without index: %s", exc)
            return None
        if not vector:
            return None
        return _pack_vector([float(x) for x in vector])

    # -- sync primitives --------------------------------------------------- #

    def get(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM store_items WHERE namespace = ? AND key = ?",
                (_ns_str(tuple(namespace)), key),
            ).fetchone()
        if row is None:
            return None
        if self._expired(row):
            self.delete(tuple(namespace), key)
            return None
        return self._row_to_item(row)

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = None,
    ) -> None:
        ns = _ns_str(tuple(namespace))
        text = _flatten(value) if self._index_enabled else None
        embedding = self._embed(text or "") if self._index_enabled else None
        created = _iso(_now())
        expires = _iso(_now() + timedelta(seconds=ttl)) if ttl else None

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at FROM store_items WHERE namespace = ? AND key = ?", (ns, key)
            ).fetchone()
            created_at = existing["created_at"] if existing else created
            conn.execute(
                "INSERT INTO store_items (namespace, key, value_json, text, embedding, "
                "created_at, updated_at, expires_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET value_json=excluded.value_json, "
                "text=excluded.text, embedding=excluded.embedding, "
                "updated_at=excluded.updated_at, expires_at=excluded.expires_at",
                (
                    ns,
                    key,
                    json.dumps(value, default=str),
                    text,
                    embedding,
                    created_at,
                    _iso(_now()),
                    expires,
                ),
            )
            conn.commit()

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM store_items WHERE namespace = ? AND key = ?",
                (_ns_str(tuple(namespace)), key),
            )
            conn.commit()

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        prefix = _ns_str(tuple(namespace_prefix))
        like = prefix + "%" if namespace_prefix else "%"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM store_items WHERE namespace LIKE ? ORDER BY updated_at DESC",
                (like,),
            ).fetchall()

        live = [r for r in rows if not self._expired(r)]
        items = [self._row_to_item(r) for r in live]

        if query:
            qvec = self._embed(query)
            if qvec is not None:
                scored: list[tuple[float, Item]] = []
                qv = _unpack_vector(qvec)
                for item, row in zip(items, live, strict=False):
                    blob = row["embedding"]
                    if not blob:
                        continue
                    similarity = _cosine(qv, _unpack_vector(blob))
                    scored.append((similarity, item))
                scored.sort(key=lambda pair: -pair[0])
                items = [
                    SearchItem(
                        namespace=i.namespace,
                        key=i.key,
                        value=i.value,
                        created_at=i.created_at,
                        updated_at=i.updated_at,
                        score=round(s, 6),
                    )
                    for s, i in scored
                ]
            else:
                terms = {t for t in query.lower().split() if len(t) > 2}
                items = [
                    i
                    for i in items
                    if any(t in _flatten(i.value).lower() for t in terms)
                ]

        # Filtering runs after scoring on purpose: zipping a filtered item list
        # against unfiltered rows would attach each item the wrong embedding and
        # rank it against the wrong neighbours.
        if filter:
            items = [i for i in items if self._matches(i.value, filter)]

        window = items[offset : offset + limit]
        if query and window and not isinstance(window[0], SearchItem):
            window = [
                SearchItem(
                    namespace=i.namespace,
                    key=i.key,
                    value=i.value,
                    created_at=i.created_at,
                    updated_at=i.updated_at,
                    score=None,
                )
                for i in window
            ]
        return list(window)

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT namespace FROM store_items").fetchall()
        out: list[tuple[str, ...]] = []
        for row in rows:
            ns = _ns_tuple(row["namespace"])
            if prefix and tuple(ns[: len(prefix)]) != tuple(prefix):
                continue
            if suffix and tuple(ns[-len(suffix) :]) != tuple(suffix):
                continue
            if max_depth is not None and len(ns) > max_depth:
                continue
            out.append(ns)
        return sorted(out)[offset : offset + limit]

    # -- BaseStore contract ------------------------------------------------ #

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key))
            elif isinstance(op, PutOp):
                self.put(op.namespace, op.key, op.value, op.index, ttl=op.ttl)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                results.append(
                    self.list_namespaces(
                        limit=op.limit, offset=op.offset, max_depth=op.max_depth
                    )
                )
            else:
                raise NotImplementedError(f"unsupported store operation: {type(op)!r}")
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return self.batch(ops)

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _expired(row: sqlite3.Row) -> bool:
        expires = row["expires_at"]
        if not expires:
            return False
        return _parse(expires) <= _now()

    @staticmethod
    def _matches(value: dict[str, Any], filters: dict[str, Any]) -> bool:
        return all(str(value.get(k)) == str(v) for k, v in filters.items())

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> Item:
        return Item(
            namespace=_ns_tuple(row["namespace"]),
            key=row["key"],
            value=json.loads(row["value_json"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    # -- agent-facing API -------------------------------------------------- #

    def remember(
        self,
        kind: str,
        key: str,
        value: dict[str, Any],
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        ttl: float | None = None,
    ) -> str:
        if user_id:
            namespace = (*USER_NS, user_id, kind)
        elif session_id:
            namespace = (*SESSION_NS, session_id, kind)
        else:
            namespace = (*FACTS_NS, kind)
        self.put(namespace, key, value, ttl=ttl)
        return ".".join(namespace)

    def recall(
        self,
        kind: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> list[Item]:
        if user_id:
            prefix = (*USER_NS, user_id, kind)
        elif session_id:
            prefix = (*SESSION_NS, session_id, kind)
        else:
            prefix = (*FACTS_NS, kind)
        return self.search(prefix, query=query, limit=limit)

    def forget(self, kind: str, key: str, *, user_id: str | None = None) -> None:
        if user_id:
            self.delete((*USER_NS, user_id, kind), key)
        else:
            self.delete((*FACTS_NS, kind), key)

    def all_items(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT namespace, key, value_json, created_at, updated_at "
                "FROM store_items ORDER BY updated_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "namespace": _ns_tuple(r["namespace"]),
                "key": r["key"],
                "value": json.loads(r["value_json"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM store_items").fetchone()["n"])


def _pack_vector(vector: list[float]) -> bytes:
    import struct

    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: bytes) -> list[float]:
    import struct

    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


_store: SqliteLongTermStore | None = None
_store_lock = threading.Lock()


def get_memory_store(settings: Settings | None = None) -> SqliteLongTermStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = SqliteLongTermStore(settings=settings)
        return _store


def reset_memory_store() -> None:
    """Drop the cached store so a test can repoint the memory database.

    Nothing to close: the store opens a connection per operation rather than
    holding one open.
    """
    global _store
    with _store_lock:
        _store = None
