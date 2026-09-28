# Document management transcript

Captured 2026-09-28T16:28:36+0530 against `http://127.0.0.1:8000`.

> The user-managed knowledge base. Verifies that a bad name is rejected rather than silently rewritten, because a rewritten name would make the following delete miss.

## 1. List before

```
$ curl -s localhost:8000/sources/documents
HTTP 200
{
  "documents": [
    {
      "name": "account-credits-and-goodwill.md",
      "bytes": 1538,
      "modified": "2026-09-27T19:27:49.254953+00:00",
      "chunks": 4
    },
    {
      "name": "billing-and-refunds.md",
      "bytes": 1775,
      "modified": "2026-09-27T19:27:41.081885+00:00",
      "chunks": 6
    },
    {
      "name": "data-export-and-offboarding.md",
      "bytes": 1702,
      "modified": "2026-09-27T19:27:52.744638+00:00",
      "chunks": 4
    },
    {
      "name": "plans-and-entitlements.md",
      "bytes": 1855,
      "modified": "2026-09-27T19:27:59.072203+00:00",
  
```

## 2. Add a document

```
$ curl -X POST localhost:8000/sources/documents \
    -H 'content-type: application/json' \
    -d '{"name": "transcript-probe-1790593116.md", "content": ...}'
HTTP 201
{
  "name": "transcript-probe-1790593116.md",
  "created": true,
  "bytes": 126,
  "removed_chunks": null,
  "indexed_chunks": 34,
  "sources": [
    "account-credits-and-goodwill.md",
    "billing-and-refunds.md",
    "data-export-and-offboarding.md",
    "plans-and-entitlements.md",
    "sla-and-escalation.md",
    "transcript-probe-1790593116.md",
    "troubleshooting-playbook.md"
  ]
}
```

## 3. Reject a traversal attempt

```
$ curl -X POST localhost:8000/sources/documents -d '{"name":"../escape.md"}'
HTTP 400
{
  "detail": "a document name must not contain path separators"
}
```

## 4. List after

```
HTTP 200
[
  {
    "name": "account-credits-and-goodwill.md",
    "bytes": 1538,
    "modified": "2026-09-27T19:27:49.254953+00:00",
    "chunks": 4
  },
  {
    "name": "billing-and-refunds.md",
    "bytes": 1775,
    "modified": "2026-09-27T19:27:41.081885+00:00",
    "chunks": 6
  },
  {
    "name": "data-export-and-offboarding.md",
    "bytes": 1702,
    "modified": "2026-09-27T19:27:52.744638+00:00",
    "chunks": 4
  },
  {
    "name": "plans-and-entitlements.md",
    "bytes": 1855,
    "modified": "2026-09-27T19:27:59.072203+00:00",
    "chunks": 8
  },
  {
    "name": "sla-and-escalation.md",
 
```

The probe appears in the listing: `{"name": "transcript-probe-1790593116.md", "bytes": 126, "modified": "2026-09-28T10:58:36.315992+00:00", "chunks": 1}`. Documents 7, chunks not reported.

## 5. Remove it

```
$ curl -X DELETE localhost:8000/sources/documents/transcript-probe-1790593116.md
HTTP 200
{
  "name": "transcript-probe-1790593116.md",
  "created": null,
  "bytes": null,
  "removed_chunks": 1,
  "indexed_chunks": 33,
  "sources": []
}
```

## Notes

- Collection held 34 chunks after the add (`created: True`) and 33 after removal, having removed 1 chunk(s) with the file.
- Mutations rebuild the whole collection rather than upserting. Chunk ids derive from content, so a rewrite cannot retire its predecessor's ids.
- The probe article is not retrievable content. It exists only to show the write path and the index accounting; every document added this way is indexed the same.
