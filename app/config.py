"""Central configuration.

Every tunable lives here and is overridable through an environment variable of the
same name. Nothing in this module encodes a business rule - those live in
``data/structured/policies.json`` and are loaded at runtime by
:mod:`app.knowledge.policy_store`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # repr=False keeps the key out of tracebacks, logs and pytest output, where
    # a stray `repr(Settings())` would otherwise publish the secret.
    groq_api_key: str = Field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-20b"
    # gpt-oss on Groq only accepts temperature=1.0; other models allow lower.
    # See ``app.llm.GroqStructuredLLM`` - it clamps this and logs when it does.
    groq_temperature: float = 1.0
    groq_max_tokens: int | None = None
    groq_base_url: str = "https://api.groq.com"

    embedding_backend: str = "auto"
    retrieval_top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150

    max_iterations: int = 4
    max_plan_steps: int = 6
    max_parallel_actions: int = 5
    max_sql_rows: int = 50

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_api_base: str = "http://127.0.0.1:8000"

    auto_approve_irreversible: bool = False

    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "knowledge" / "kb"

    @property
    def structured_dir(self) -> Path:
        return self.data_dir / "structured"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "support.db"

    @property
    def checkpoint_db_path(self) -> Path:
        """Separate file for LangGraph checkpoints.

        Kept apart from ``support.db`` so that checkpoint bookkeeping never
        interferes with the seeded operational data, and so the operational
        database can be reset without discarding a pending approval.
        """
        return self.data_dir / "checkpoints.db"

    @property
    def memory_db_path(self) -> Path:
        return self.data_dir / "memory.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"

    @property
    def policies_path(self) -> Path:
        return self.structured_dir / "policies.json"

    @property
    def catalog_path(self) -> Path:
        return self.structured_dir / "catalog.csv"

    @property
    def has_llm_key(self) -> bool:
        return bool(self.groq_api_key.strip())

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.chroma_dir, self.outbox_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
