"""Shared fixtures.

Every test runs against a throwaway ``DATA_DIR`` seeded from the same source
files as production, so tests can freely write tickets, notes and credits
without touching ``data/support.db``. The hashing embedding backend is forced so
the suite needs no model download and no network.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.knowledge import sql_store  # noqa: E402
from app.knowledge.catalog_store import load_catalog  # noqa: E402
from app.knowledge.embeddings import reset_embeddings_cache  # noqa: E402
from app.knowledge.policy_store import load_policies  # noqa: E402
from app.knowledge.vector_store import index_kb, reset_vector_store  # noqa: E402
from app.memory import reset_memory_store  # noqa: E402


def _reset_caches() -> None:
    """Drop every process-global cache that binds to a data directory."""
    get_settings.cache_clear()
    reset_vector_store()
    reset_embeddings_cache()
    reset_memory_store()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """An isolated, seeded copy of the data directory."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("AUTO_APPROVE_IRREVERSIBLE", "0")
    _reset_caches()

    settings = get_settings()
    settings.ensure_dirs()

    # copy the real source files in, then seed from them
    for folder in ("knowledge", "structured"):
        source = PROJECT_ROOT / "data" / folder
        target = settings.data_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)

    sql_store.init_db(settings)
    sql_store.seed_from_csv(settings=settings)
    index_kb(force=True, settings=settings)

    # Reload the JSON/CSV caches only after the files are in place; forcing
    # earlier would raise on a directory that does not exist yet.
    load_policies(force=True, settings=settings)
    load_catalog(force=True, settings=settings)

    yield settings

    _reset_caches()


@pytest.fixture
def live_groq() -> bool:
    """True when a real key is configured, so live tests can skip cleanly."""
    return bool(get_settings().has_llm_key)
