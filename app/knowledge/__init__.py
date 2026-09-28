"""Knowledge source access layer.

Four independent backends, deliberately kept separate so the retrieval node can
fan out across them in parallel and so a failure in one source does not take the
others down:

* :mod:`app.knowledge.vector_store` - Markdown KB over Chroma
* :mod:`app.knowledge.sql_store`   - operational SQLite (read-only SQL)
* :mod:`app.knowledge.policy_store` - business policy JSON
* :mod:`app.knowledge.catalog_store` - product catalog CSV
"""
