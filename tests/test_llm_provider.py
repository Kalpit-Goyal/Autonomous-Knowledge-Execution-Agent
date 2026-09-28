"""Provider adapter tests.

These pin the parts of the Groq adapter that are easy to get subtly wrong and
that no offline end-to-end test would catch: the temperature clamp for models
Groq pins to a single value, and the unpack of an ``include_raw`` response.
No network and no key required.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm import (
    STRUCTURED_METHODS,
    GroqStructuredLLM,
    LLMRateLimited,
    LLMUnavailable,
    _effective_temperature,
    _unpack,
)
from app.schemas import Intake


def _settings(**overrides) -> Settings:
    base = {"groq_api_key": "test-key-not-real", "data_dir": "data"}
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# key handling
# --------------------------------------------------------------------------- #


def test_missing_key_is_reported_with_the_right_variable_name(sandbox):
    with pytest.raises(LLMUnavailable) as excinfo:
        GroqStructuredLLM(_settings(groq_api_key=""))
    message = str(excinfo.value)
    assert "GROQ_API_KEY" in message
    assert "XAI_API_KEY" not in message, "must not name the previous provider"


def test_blank_key_counts_as_missing(sandbox):
    assert _settings(groq_api_key="   ").has_llm_key is False
    assert _settings(groq_api_key="gsk-real").has_llm_key is True


# --------------------------------------------------------------------------- #
# temperature
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"])
def test_gpt_oss_is_pinned_to_temperature_one(sandbox, model):
    assert _effective_temperature(_settings(groq_model=model, groq_temperature=1.0)) == 1.0
    assert _effective_temperature(_settings(groq_model=model, groq_temperature=0.1)) == 1.0
    assert _effective_temperature(_settings(groq_model=model, groq_temperature=0.9)) == 1.0


def test_other_models_keep_the_requested_temperature(sandbox):
    settings = _settings(groq_model="llama-3.3-70b-versatile", groq_temperature=0.2)
    assert _effective_temperature(settings) == 0.2


def test_default_temperature_is_safe_for_the_default_model(sandbox):
    """The default model is gpt-oss, so the default temperature must already be 1.0."""
    from app.config import Settings

    # The declared field default, so a developer's local .env cannot mask it.
    assert Settings.model_fields["groq_model"].default == "openai/gpt-oss-20b"
    assert _effective_temperature(_settings(groq_temperature=1.0)) == 1.0
    assert _effective_temperature(_settings()) == 1.0


# --------------------------------------------------------------------------- #
# include_raw unpacking
# --------------------------------------------------------------------------- #


class _FakeRaw:
    def __init__(self, tin: int, tout: int) -> None:
        self.usage_metadata = {"input_tokens": tin, "output_tokens": tout}


def test_unpack_recovers_tokens_from_the_raw_response(sandbox):
    """A parsed pydantic object carries no usage, so the counts come from `raw`."""
    payload = Intake(summary="billing question", customer_id="C-1004", ticket_id="T-5005")
    result, tin, tout = _unpack(
        {"raw": _FakeRaw(120, 45), "parsed": payload, "parsing_error": None},
        Intake,
        "function_calling",
    )
    assert result is payload
    assert (tin, tout) == (120, 45)


def test_unpack_validates_a_dict_payload(sandbox):
    result, _, _ = _unpack(
        {"raw": _FakeRaw(1, 1), "parsed": {"summary": "s", "customer_id": "C-1001"}},
        Intake,
        "json_schema",
    )
    assert isinstance(result, Intake)
    assert result.customer_id == "C-1001"


def test_unpack_accepts_a_bare_object(sandbox):
    """Falls back gracefully if a provider ignores include_raw."""
    payload = Intake(summary="refund question", customer_id="C-1002")
    result, _, _ = _unpack(payload, Intake, "function_calling")
    assert result is payload


def test_unpack_raises_when_the_model_returned_nothing(sandbox):
    with pytest.raises(ValueError) as excinfo:
        _unpack(
            {"raw": _FakeRaw(0, 0), "parsed": None, "parsing_error": "bad json"},
            Intake,
            "json_mode",
        )
    assert "bad json" in str(excinfo.value)


def test_unpack_rejects_a_wrong_type(sandbox):
    with pytest.raises(TypeError):
        _unpack({"raw": _FakeRaw(0, 0), "parsed": "not an intake"}, Intake, "json_mode")


# --------------------------------------------------------------------------- #
# method fallback order
# --------------------------------------------------------------------------- #


def test_structured_methods_start_with_tool_calling_and_end_with_json(sandbox):
    """Tool calling is the most reliable on Groq; json_mode is the last resort."""
    assert STRUCTURED_METHODS[0] == "function_calling"
    assert STRUCTURED_METHODS[-1] == "json_mode"
    assert len(set(STRUCTURED_METHODS)) == 3


def test_falls_back_to_the_next_method_when_one_is_rejected(sandbox, monkeypatch):
    """A model that refuses json_schema must not fail the whole run."""
    calls: list[str] = []

    class _FakeChat:
        def with_structured_output(self, schema, method=None, include_raw=False):
            calls.append(method)
            if method == "function_calling":
                raise RuntimeError("tool calling is not supported for this model")

            class _Runnable:
                def invoke(self, messages):
                    payload = Intake(summary="s", customer_id="C-1003")
                    return {"raw": _FakeRaw(7, 3), "parsed": payload}

            return _Runnable()

    llm = object.__new__(GroqStructuredLLM)
    llm.settings = _settings()
    llm.model_name = "openai/gpt-oss-120b"
    llm._chat = _FakeChat()

    result, tin, tout = llm.structured(
        Intake, system="s", human="C-1003 asked about billing", node="unit"
    )

    assert result.customer_id == "C-1003"
    assert (tin, tout) == (7, 3)
    assert calls == ["function_calling", "json_schema"]


def test_exhausting_every_method_names_them_in_the_error(sandbox, monkeypatch):
    monkeypatch.setattr("app.llm.time.sleep", lambda _s: None)

    class _FakeChat:
        def with_structured_output(self, schema, method=None, include_raw=False):
            class _Runnable:
                def invoke(self, messages):
                    raise RuntimeError(f"{method} exploded")

            return _Runnable()

    llm = object.__new__(GroqStructuredLLM)
    llm.settings = _settings()
    llm.model_name = "openai/gpt-oss-120b"
    llm._chat = _FakeChat()

    with pytest.raises(RuntimeError) as excinfo:
        llm.structured(Intake, system="s", human="h", node="unit")

    message = str(excinfo.value)
    assert "unit" in message
    for method in STRUCTURED_METHODS:
        assert method in message


def test_settings_repr_never_leaks_the_api_key():
    """A settings dump in a traceback must not publish the secret.

    A plain ``str`` field shows up in every pydantic repr, so an assertion
    failure that prints the fixture leaks the key into CI logs.
    """
    settings = Settings(groq_api_key="gsk_supersecretvalue")
    assert settings.groq_api_key == "gsk_supersecretvalue", "the key must still be usable"
    assert "supersecretvalue" not in repr(settings)
    assert "groq_api_key" not in repr(settings)
    assert "supersecretvalue" not in f"{settings!r}"


def test_text_does_not_forward_extra_kwargs_to_the_chat_client():
    """ChatGroq forwards unknown kwargs to Completions.create(), which 400s.

    `text()` used to pass `node=...` purely for logging, which made every
    responder call fail against the real API with
    "unexpected keyword argument 'node'". Offline fakes accepted **kwargs and
    hid it, so this fake rejects them the way the real client does.
    """
    seen: list[object] = []

    class _StrictResp:
        content = "Growth is $499 per seat per month."
        usage_metadata = {"input_tokens": 11, "output_tokens": 5}

    class _StrictChat:
        def invoke(self, messages, *args, **kwargs):
            seen.append(kwargs)
            if kwargs:
                raise TypeError(
                    f"Completions.create() got an unexpected keyword argument "
                    f"'{next(iter(kwargs))}'"
                )
            return _StrictResp()

    llm = object.__new__(GroqStructuredLLM)
    llm.settings = _settings()
    llm.model_name = "openai/gpt-oss-120b"
    llm._chat = _StrictChat()

    text, tin, tout = llm.text(system="s", human="h", node="respond")

    assert text == "Growth is $499 per seat per month."
    assert (tin, tout) == (11, 5)
    assert all(kwargs == {} for kwargs in seen), "no kwargs may reach ChatGroq.invoke"


def test_rate_limit_is_not_retried_across_every_method():
    """A spent token budget must fail fast, not burn nine more calls.

    Each live test drives a full graph, so a 429 that is retried 3 methods x 3
    attempts compounds the quota loss instead of stopping.
    """
    calls: list[str] = []

    def _rate_limited():
        return RuntimeError(
            "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
            "`openai/gpt-oss-120b` ... on tokens per day (TPD): Limit 200000'}}"
        )

    class _FakeChat:
        def with_structured_output(self, schema, method=None, include_raw=False):
            calls.append(method)

            class _Runnable:
                def invoke(self, messages):
                    calls.append(f"{method}:invoke")
                    raise _rate_limited()

            return _Runnable()

    llm = object.__new__(GroqStructuredLLM)
    llm.settings = _settings()
    llm.model_name = "openai/gpt-oss-120b"
    llm._chat = _FakeChat()

    with pytest.raises(LLMRateLimited) as excinfo:
        llm.structured(Intake, system="s", human="h", node="intake")

    assert "intake" in str(excinfo.value)
    invocations = [c for c in calls if c.endswith(":invoke")]
    assert len(invocations) == 1, f"a 429 must not be retried or fall through: {calls}"
    assert "json_schema" not in calls and "json_mode" not in calls


def test_text_rate_limit_fails_fast():
    calls: list[int] = []

    class _FakeChat:
        def invoke(self, messages):
            calls.append(1)
            raise RuntimeError("Error code: 429 - rate limit reached")

    llm = object.__new__(GroqStructuredLLM)
    llm.settings = _settings()
    llm.model_name = "openai/gpt-oss-120b"
    llm._chat = _FakeChat()

    with pytest.raises(LLMRateLimited):
        llm.text(system="s", human="h", node="respond")

    assert len(calls) == 1, f"a 429 must not be retried, got {len(calls)} calls"
