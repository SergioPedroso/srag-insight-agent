from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from srag_agent.audit import AuditTrail
from srag_agent.llm.gemini import GeminiProvider


class OverloadedThenOk:
    def __init__(self, failures: int):
        self.failures = failures
        self.models_called = []

    def generate_content(self, *, model, contents, config):
        self.models_called.append(model)
        if len(self.models_called) <= self.failures:
            raise genai_errors.ServerError(503, {"error": {"message": "high demand"}})
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(role="model", parts=[types.Part.from_text(text="ok")]),
                    finish_reason=SimpleNamespace(name="STOP"),
                )
            ],
            usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
            model_version=model,
            response_id="r",
        )


def _provider(tmp_path, models, fallback="lite"):
    audit = AuditTrail(tmp_path)
    provider = GeminiProvider(
        "main",
        audit,
        api_key=None,
        max_tokens=100,
        fallback_model=fallback,
        client=SimpleNamespace(models=models),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text="oi")])]
    return provider, audit, contents


def test_switches_to_fallback_model_when_overloaded(tmp_path):
    models = OverloadedThenOk(failures=1)
    provider, audit, contents = _provider(tmp_path, models)

    provider.generate("teste", "sistema", contents, {})
    provider.generate("teste", "sistema", contents, {})

    assert models.models_called == ["main", "lite", "lite"]  # a troca é mantida
    fallback_events = [e for e in audit.read() if e["event"] == "llm_fallback"]
    assert len(fallback_events) == 1 and fallback_events[0]["to_model"] == "lite"


def test_raises_when_fallback_also_fails(tmp_path):
    provider, _, contents = _provider(tmp_path, OverloadedThenOk(failures=2))
    with pytest.raises(genai_errors.ServerError):
        provider.generate("teste", "sistema", contents, {})
