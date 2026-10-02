"""Camada de LLM independente de provedor (Gemini ou Claude), escolhida pela configuração."""

from typing import Any

from srag_agent.audit import AuditTrail
from srag_agent.config import Settings
from srag_agent.llm.base import (
    AssistantTurn,
    LLMProvider,
    LLMRefusalError,
    ToolCall,
    ToolDefinition,
    ToolOutcome,
    ToolSession,
)

__all__ = [
    "AssistantTurn",
    "LLMProvider",
    "LLMRefusalError",
    "MissingAPIKeyError",
    "ToolCall",
    "ToolDefinition",
    "ToolOutcome",
    "ToolSession",
    "create_provider",
]


class MissingAPIKeyError(RuntimeError):
    pass


def _secret(value) -> str | None:
    return value.get_secret_value() if value else None


def create_provider(
    settings: Settings, audit: AuditTrail, client: Any | None = None
) -> LLMProvider:
    """Instancia o provedor configurado em `SRAG_LLM_PROVIDER`."""
    if settings.llm_provider == "gemini":
        from srag_agent.llm.gemini import GeminiProvider

        api_key = _secret(settings.gemini_api_key)
        if client is None and not api_key:
            raise MissingAPIKeyError("Defina GEMINI_API_KEY no arquivo .env.")
        return GeminiProvider(
            settings.model_name,
            audit,
            api_key=api_key,
            max_tokens=settings.llm_max_tokens,
            fallback_model=settings.fallback_model_name,
            client=client,
        )

    from srag_agent.llm.claude import ClaudeProvider

    api_key = _secret(settings.anthropic_api_key)
    if client is None and not api_key:
        raise MissingAPIKeyError("Defina ANTHROPIC_API_KEY no arquivo .env.")
    return ClaudeProvider(
        settings.model_name,
        audit,
        api_key=api_key,
        effort=settings.llm_effort,
        max_tokens=settings.llm_max_tokens,
        client=client,
    )
