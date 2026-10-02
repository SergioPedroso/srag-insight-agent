"""Adaptador da Claude (Anthropic), via SDK oficial.

Usa o fallback de modelo do próprio servidor quando uma requisição é recusada pelos
classificadores de segurança, e schemas `strict` nas tools.
"""

from typing import Any

import anthropic

from srag_agent.audit import AuditTrail
from srag_agent.llm.base import (
    AssistantTurn,
    LLMProvider,
    LLMRefusalError,
    ModelT,
    ToolCall,
    ToolDefinition,
    ToolOutcome,
    ToolSession,
)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(
        self,
        model: str,
        audit: AuditTrail,
        *,
        api_key: str | None,
        effort: str,
        max_tokens: int,
        client: Any | None = None,
    ):
        super().__init__(model, audit)
        self._client = client or anthropic.Anthropic(api_key=api_key)
        self._effort = effort
        self._max_tokens = max_tokens

    def _params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "output_config": {"effort": self._effort},
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def _record(self, purpose: str, system: str, messages: list, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.record_call(
            purpose,
            system=system,
            payload=messages,
            served_model=getattr(response, "model", None),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            request_id=getattr(response, "_request_id", None),
        )
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMRefusalError(f"Requisição recusada pelo modelo (categoria: {category})")

    def create(self, purpose: str, system: str, messages: list, tools: list) -> Any:
        response = self._client.beta.messages.create(
            **self._params(), system=system, messages=messages, tools=tools
        )
        self._record(purpose, system, messages, response)
        return response

    def start_tool_session(
        self, purpose: str, system: str, user_message: str, tools: list[ToolDefinition]
    ) -> ToolSession:
        return _ClaudeToolSession(self, purpose, system, user_message, tools)

    def structured(
        self, purpose: str, system: str, user_message: str, output_format: type[ModelT]
    ) -> ModelT:
        messages = [{"role": "user", "content": user_message}]
        response = self._client.beta.messages.parse(
            **self._params(), system=system, messages=messages, output_format=output_format
        )
        self._record(purpose, system, messages, response)
        if response.parsed_output is None:
            raise ValueError("O modelo não retornou uma saída estruturada válida.")
        return response.parsed_output


class _ClaudeToolSession(ToolSession):
    def __init__(
        self,
        provider: ClaudeProvider,
        purpose: str,
        system: str,
        user_message: str,
        tools: list[ToolDefinition],
    ):
        self._provider = provider
        self._purpose = purpose
        self._system = system
        self._messages: list = [{"role": "user", "content": user_message}]
        self._tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "strict": True,
            }
            for t in tools
        ]

    def next_turn(self) -> AssistantTurn:
        response = self._provider.create(self._purpose, self._system, self._messages, self._tools)
        # O conteúdo volta inalterado (inclui blocos de raciocínio exigidos pela API).
        self._messages.append({"role": "assistant", "content": response.content})
        return AssistantTurn(
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ],
        )

    def add_tool_results(self, outcomes: list[ToolOutcome]) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": o.call.id,
                        "content": o.content,
                        "is_error": o.is_error,
                    }
                    for o in outcomes
                ],
            }
        )
