# Streaming transport transcript

Captured 2026-09-28T16:28:03+0530 against `http://127.0.0.1:8000`.

> This is a real request against a real provider that failed on a daily token limit, so it records the transport contract and the in-band error path. A successful turn is in 04.

## 1. Stream a question

```
$ curl -N -X POST localhost:8000/chat/stream -d '{"message":"..."}'
HTTP 200
content-type: text/event-stream; charset=utf-8
cache-control: no-cache, no-transform
x-accel-buffering: no

  [error] data: {"event": "error", "session_id": "SESS-e6aaf07f9d", "error": "LLMRateLimited: intake: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_***` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used
  [stream closed at 0.25s]
```

## The error is in-band, not a transport failure

This run is a genuine failure worth showing: the provider rejected the call after the response had already committed to `200 text/event-stream`, so the only way to report it is a frame.

```
{
  "event": "error",
  "session_id": "SESS-e6aaf07f9d",
  "error": "LLMRateLimited: intake: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_***` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199790, Requested 1402. Please try again in 8m34.943999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/... 'type': 'tokens', 'code': 'rate_limit_exceeded'}}",
  "elapsed_ms": 222
}
```

The client cannot distinguish this from a transport error by status code alone, which is why the UI watches the last frame and not just the response code.

The 429 is a **daily** quota, not a per-minute one, and it is only detectable at the point of call. This run gave up at the first node (0 node events, 0.25s); an earlier one retried through 27 node events before surfacing the same limit. Worth knowing before pointing this at a real queue.

## Frame counts

| Event | Frames |
| --- | --- |
| `error` | 1 |
| `start` | 1 |

## First occurrences

```
start -> error
```

## Notes

- `cache-control: no-cache, no-transform` and `x-accel-buffering: no` are both required. Without the second, a reverse proxy will buffer the whole body and the stream arrives as one blob at the end.
- The `done` frame carries the same ChatResponse as the blocking `/chat` endpoint. A test pins that the concatenated token frames equal it.
