"""Rebuild the Chroma index from the markdown knowledge base.

    python scripts/ingest.py            # index only if the sources changed
    python scripts/ingest.py --force    # always rebuild
    python scripts/ingest.py --query "refund window for annual plans"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.knowledge.vector_store import collection_stats, index_kb, search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ingest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    parser.add_argument("--query", help="run a sample retrieval after indexing")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()

    result = index_kb(force=args.force, settings=settings)
    logger.info("index: %s", json.dumps(result, default=str))
    logger.info("stats: %s", json.dumps(collection_stats(settings), default=str))

    if args.query:
        logger.info("query: %s", args.query)
        for ev in search(args.query, top_k=args.top_k, settings=settings):
            logger.info("  [%.3f] %s", ev.score, ev.citation)
            logger.info("        %s", ev.snippet[:160].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
