"""Cliente da Claude com auditoria de cada chamada.

Toda requisição registra: modelo solicitado e efetivo, motivo de parada, uso de tokens,
hash do prompt de sistema e das mensagens. Recusas por política de segurança são
tratadas explicitamente (com fallback de modelo feito pelo próprio servidor).
"""

import json
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from srag_agent.audit import AuditTrail, sha256
from srag_agent.config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMRefusalError(RuntimeError):
    """O modelo (e o fallback) recusaram a requisição."""


class LLMClient:
    def __init__(self, settings: Settings, audit: AuditTrail, client: Any | None = None):
        self._settings = settings
        self._audit = audit
        if client is None:
            api_key = settings.anthropic_api_key
            client = anthropic.Anthropic(api_key=api_key.get_secret_value() if api_key else None)
        self._client = client

    def _common_params(self) -> dict[str, Any]:
        return {
            "model": self._settings.llm_model,
            "max_tokens": self._settings.llm_max_tokens,
            "output_config": {"effort": self._settings.llm_effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def _record(self, purpose: str, system: str, messages: list, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self._audit.record(
            "llm_call",
            purpose,
            requested_model=self._settings.llm_model,
            served_model=getattr(response, "model", None),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            system_prompt_sha256=sha256(system),
            messages_sha256=sha256(json.dumps(messages, default=str, ensure_ascii=False)),
            request_id=getattr(response, "_request_id", None),
        )
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(
                f"Requisição recusada pelo modelo (categoria: {getattr(details, 'category', None)})"
            )

    def run_with_tools(self, purpose: str, system: str, messages: list, tools: list) -> Any:
        response = self._client.beta.messages.create(
            **self._common_params(), system=system, messages=messages, tools=tools
        )
        self._record(purpose, system, messages, response)
        return response

    def structured(
        self, purpose: str, system: str, messages: list, output_format: type[ModelT]
    ) -> ModelT:
        response = self._client.beta.messages.parse(
            **self._common_params(),
            system=system,
            messages=messages,
            output_format=output_format,
        )
        self._record(purpose, system, messages, response)
        if response.parsed_output is None:
            raise ValueError("O modelo não retornou uma saída estruturada válida.")
        return response.parsed_output
