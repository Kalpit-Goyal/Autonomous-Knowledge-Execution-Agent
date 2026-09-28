"""LLM access.

The graph never touches a model class directly. It asks for a *typed* result
through :meth:`StructuredLLM.structured` and for prose through
:meth:`StructuredLLM.text`, which keeps every node testable against
:class:`ScriptedLLM` with no network and no API key.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from pydantic import BaseModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 3
RETRY_SLEEP = 2.0

# Groq hosts the open-weight gpt-oss models, which only accept temperature=1.0
# and reject any other value with a 400. Sampling settings are clamped for them
# rather than letting a 400 surface from deep inside a node.
FIXED_TEMPERATURE_MODELS = ("gpt-oss",)

# Ordered by preference. Groq exposes three different ways to constrain output
# and support varies by model, so a failure on one is retried against the next
# instead of failing the run.
STRUCTURED_METHODS = ("function_calling", "json_schema", "json_mode")

_JSON_MODE_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else - no prose, no "
    "markdown fences, no explanation. It must validate against this JSON schema:\n"
    "{schema}"
)


class LLMUnavailable(RuntimeError):
    pass


class LLMRateLimited(RuntimeError):
    """Raised when Groq refuses on quota/throughput grounds.

    Distinguished from other errors because retrying is pointless and expensive:
    the token budget is gone, so a full 3-method x 3-attempt sweep would issue
    nine more doomed calls and deepen the problem.
    """


def _is_rate_limited(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
        or "429" in text
        or "tokens per day" in text
    )


def _set_usage(usage: dict[str, int] | None, tin: int, tout: int) -> None:
    if usage is not None:
        usage["in"] = tin
        usage["out"] = tout


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    tin = int(usage.get("input_tokens") or 0)
    tout = int(usage.get("output_tokens") or 0)
    if not tin and not tout:
        meta = getattr(response, "response_metadata", {}) or {}
        raw = meta.get("token_usage") or {}
        tin = int(raw.get("prompt_tokens") or 0)
        tout = int(raw.get("completion_tokens") or 0)
    return tin, tout


def _effective_temperature(settings: Settings) -> float:
    """Clamp sampling for models Groq pins to a single value."""
    wanted = settings.groq_temperature
    if any(marker in settings.groq_model for marker in FIXED_TEMPERATURE_MODELS):
        if wanted != 1.0:
            logger.warning(
                "%s only accepts temperature=1.0 on Groq; ignoring GROQ_TEMPERATURE=%s",
                settings.groq_model,
                wanted,
            )
        return 1.0
    return wanted


def _unpack[T: BaseModel](raw_result: Any, schema: type[T], method: str) -> tuple[T, int, int]:
    """Pull the validated object and token counts out of a Groq response.

    ``include_raw=True`` returns ``{"raw": AIMessage, "parsed": ...}`` so the
    token counts survive - a bare pydantic object carries no usage metadata, so
    counting tokens from the parsed result alone always reports zero.
    """
    tin = tout = 0
    result = raw_result
    if isinstance(raw_result, dict) and "parsed" in raw_result:
        result = raw_result.get("parsed")
        tin, tout = _usage_tokens(raw_result.get("raw"))
        if result is None and raw_result.get("parsing_error"):
            raise ValueError(
                f"{method}: model returned no usable object: {raw_result['parsing_error']}"
            )
    else:
        tin, tout = _usage_tokens(raw_result)

    if isinstance(result, dict):
        result = schema.model_validate(result)
    if not isinstance(result, schema):
        raise TypeError(
            f"{method}: expected {schema.__name__}, got {type(result).__name__}"
        )
    return result, tin, tout


class GroqStructuredLLM:
    """Thin wrapper over ``ChatGroq`` with retries and token accounting."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.settings = settings
        self.model_name = settings.groq_model
        if not settings.has_llm_key:
            raise LLMUnavailable(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq key, "
                "or set AGENT_LLM_MODE=fake to run without a model."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("langchain-groq is not installed") from exc

        self._chat = ChatGroq(
            model=settings.groq_model,
            temperature=_effective_temperature(settings),
            max_tokens=settings.groq_max_tokens,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            max_retries=2,
            timeout=120,
        )

    def structured(
        self, schema: type[T], *, system: str, human: str, node: str = ""
    ) -> tuple[T, int, int]:
        """Return a validated ``schema`` instance, trying each Groq method in turn."""
        last: Exception | None = None
        for method in STRUCTURED_METHODS:
            system_prompt = system
            if method == "json_mode":
                # json_mode guarantees parseable JSON but not the shape, so the
                # schema has to be stated in the prompt.
                system_prompt = system + "\n\n" + _JSON_MODE_INSTRUCTION.format(
                    schema=schema.model_json_schema()
                )
            try:
                runnable = self._chat.with_structured_output(
                    schema, method=method, include_raw=True
                )
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limited(exc):
                    raise LLMRateLimited(f"{node}: {exc}") from exc
                logger.warning("%s: method %s unavailable: %s", node, method, exc)
                last = exc
                continue

            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    raw_result = runnable.invoke(
                        [("system", system_prompt), ("human", human)]
                    )
                    result, tin, tout = _unpack(raw_result, schema, method)
                    return result, tin, tout
                except Exception as exc:  # noqa: BLE001
                    if _is_rate_limited(exc):
                        raise LLMRateLimited(f"{node}: {exc}") from exc
                    last = exc
                    logger.warning(
                        "%s structured call failed via %s (attempt %d/%d): %s",
                        node or schema.__name__,
                        method,
                        attempt,
                        MAX_ATTEMPTS,
                        exc,
                    )
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(RETRY_SLEEP * attempt)

        label = node or schema.__name__
        tried = ", ".join(STRUCTURED_METHODS)
        raise RuntimeError(
            f"{label} failed after {MAX_ATTEMPTS} attempts per method [{tried}]: {last}"
        )

    def text(self, *, system: str, human: str, node: str = "") -> tuple[str, int, int]:
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                # No extra kwargs: ChatGroq forwards unknown ones straight to
                # Completions.create(), which rejects them.
                response = self._chat.invoke([("system", system), ("human", human)])
                content = response.content
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content), *_usage_tokens(response)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limited(exc):
                    raise LLMRateLimited(f"{node}: {exc}") from exc
                last = exc
                logger.warning(
                    "%s text call failed (attempt %d/%d): %s",
                    node or "text",
                    attempt,
                    MAX_ATTEMPTS,
                    exc,
                )
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_SLEEP * attempt)
        raise RuntimeError(f"{node} failed after {MAX_ATTEMPTS} attempts: {last}")

    def stream_text(
        self, *, system: str, human: str, node: str = "", usage: dict[str, int] | None = None
    ) -> Iterator[str]:
        """Yield answer text as the provider produces it.

        ``usage`` is filled in once the stream finishes. Groq only reports token
        counts on the final chunk, so the counts cannot be returned as a tuple
        from a generator; the caller reads them from this dict afterwards.
        """
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            parts: list[str] = []
            try:
                for chunk in self._chat.stream([("system", system), ("human", human)]):
                    # Usage is read before the text check: providers routinely
                    # report the final token counts on an empty trailing chunk.
                    counts = _usage_tokens(chunk)
                    if counts != (0, 0):
                        _set_usage(usage, *counts)
                    content = getattr(chunk, "content", "")
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    text = str(content or "")
                    if not text:
                        continue
                    parts.append(text)
                    yield text
                if parts:
                    return
                last = RuntimeError("the provider streamed no text")
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limited(exc):
                    raise LLMRateLimited(f"{node}: {exc}") from exc
                last = exc
                logger.warning(
                    "%s stream failed (attempt %d/%d): %s",
                    node or "text",
                    attempt,
                    MAX_ATTEMPTS,
                    exc,
                )
                if attempt >= MAX_ATTEMPTS:
                    break
                time.sleep(RETRY_SLEEP * attempt)
        raise RuntimeError(f"{node} failed to stream after {MAX_ATTEMPTS} attempts: {last}")


Responder = Callable[[type[BaseModel], str, str, str], BaseModel]
TextResponder = Callable[[str, str, str], str]


class ScriptedLLM:
    """Offline stand-in used by the test suite.

    The responder receives the requested schema plus the system and human
    prompts and returns a fully-formed instance, so a test can script an entire
    multi-node run without a model or a key.
    """

    def __init__(
        self,
        responder: Responder,
        text_responder: TextResponder | None = None,
        name: str = "scripted",
    ) -> None:
        self._responder = responder
        self._text_responder = text_responder or (lambda system, human, node: "(scripted reply)")
        self.model_name = name
        self.calls: list[tuple[str, str]] = []

    def structured(
        self, schema: type[T], *, system: str, human: str, node: str = ""
    ) -> tuple[T, int, int]:
        self.calls.append((node or schema.__name__, human[:80]))
        result = self._responder(schema, system, human, node or schema.__name__)
        if isinstance(result, dict):
            result = schema.model_validate(result)
        if not isinstance(result, schema):
            raise TypeError(
                f"scripted responder returned {type(result).__name__}, expected {schema.__name__}"
            )
        return result, 0, 0

    def text(self, *, system: str, human: str, node: str = "") -> tuple[str, int, int]:
        self.calls.append((node, human[:80]))
        return self._text_responder(system, human, node), 0, 0

    def stream_text(
        self, *, system: str, human: str, node: str = "", usage: dict[str, int] | None = None
    ) -> Iterator[str]:
        """Same text as ``text()``, split into word-sized deltas.

        Gives the offline suite a real token stream to assert against without a
        provider.
        """
        full = self.text(system=system, human=human, node=node)[0]
        for word in full.split(" "):
            yield word + " "
        _set_usage(usage, 0, 0)


def build_llm(settings: Settings | None = None) -> GroqStructuredLLM | ScriptedLLM:
    """Return the LLM for this process.

    ``AGENT_LLM_MODE=fake`` yields a ScriptedLLM that raises if a node actually
    tries to reason, so a misconfigured environment fails loudly instead of
    silently returning placeholder text.
    """
    settings = settings or get_settings()
    mode = os.getenv("AGENT_LLM_MODE", "groq").strip().lower()
    if mode == "fake":
        def _refuse(schema, system, human, node):
            raise LLMUnavailable(
                f"AGENT_LLM_MODE=fake but node '{node}' requested a {schema.__name__}. "
                "Set GROQ_API_KEY in .env to run against Groq."
            )

        return ScriptedLLM(_refuse, name="fake")
    return GroqStructuredLLM(settings)
