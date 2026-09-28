"""Load the structured CSV sources into SQLite and index the knowledge base.

    python scripts/seed.py            # create/refresh the ops database
    python scripts/ingest.py --force  # rebuild the Chroma index from markdown
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.knowledge import sql_store  # noqa: E402
from app.knowledge.vector_store import collection_stats, index_kb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete existing rows first")
    parser.add_argument("--no-index", action="store_true", help="skip the Chroma index build")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()

    logger.info("seeding ops database from %s", settings.structured_dir)
    counts = sql_store.seed_from_csv(reset=args.reset, settings=settings)
    for table, n in counts.items():
        logger.info("  %-14s %4d rows", table, n)

    if not args.no_index:
        logger.info("indexing knowledge base from %s", settings.kb_dir)
        result = index_kb(force=True, settings=settings)
        logger.info("  %s", json.dumps(result, default=str))

    logger.info("database: %s", settings.db_path)
    logger.info("index:    %s", settings.chroma_dir)
    logger.info("stats:    %s", json.dumps(collection_stats(settings), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
