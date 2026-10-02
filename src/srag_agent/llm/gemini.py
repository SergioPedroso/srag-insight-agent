"""Adaptador do Google Gemini, via SDK oficial `google-genai`.

Usa chamada manual de funções (o agente executa as tools e audita cada uma) e saída
estruturada com schema Pydantic. O conteúdo devolvido pelo modelo é reenviado sem
alterações, preservando as assinaturas de raciocínio exigidas pela API.
"""

import json
import uuid
from typing import Any

from google import genai
from google.genai import types

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

BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}


class GeminiProvider(LLMProvider):
    provider_name = "google"

    def __init__(
        self,
        model: str,
        audit: AuditTrail,
        *,
        api_key: str | None,
        max_tokens: int,
        client: Any | None = None,
    ):
        super().__init__(model, audit)
        self._client = client or genai.Client(api_key=api_key)
        self._max_tokens = max_tokens

    def generate(self, purpose: str, system: str, contents: list, config: dict[str, Any]) -> Any:
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=self._max_tokens, **config
            ),
        )
        candidate = response.candidates[0] if response.candidates else None
        finish = getattr(getattr(candidate, "finish_reason", None), "name", None)
        usage = response.usage_metadata
        self.record_call(
            purpose,
            system=system,
            payload=[c.model_dump(mode="json", exclude_none=True) for c in contents],
            served_model=getattr(response, "model_version", None),
            stop_reason=finish,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            request_id=getattr(response, "response_id", None),
        )
        if candidate is None or finish in BLOCKED_FINISH_REASONS:
            raise LLMRefusalError(f"Resposta bloqueada pelo provedor (motivo: {finish})")
        return response

    def start_tool_session(
        self, purpose: str, system: str, user_message: str, tools: list[ToolDefinition]
    ) -> ToolSession:
        return _GeminiToolSession(self, purpose, system, user_message, tools)

    def structured(
        self, purpose: str, system: str, user_message: str, output_format: type[ModelT]
    ) -> ModelT:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]
        response = self.generate(
            purpose,
            system,
            contents,
            {"response_mime_type": "application/json", "response_schema": output_format},
        )
        if isinstance(response.parsed, output_format):
            return response.parsed
        return output_format.model_validate_json(response.text)


class _GeminiToolSession(ToolSession):
    def __init__(
        self,
        provider: GeminiProvider,
        purpose: str,
        system: str,
        user_message: str,
        tools: list[ToolDefinition],
    ):
        self._provider = provider
        self._purpose = purpose
        self._system = system
        self._contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        ]
        self._config = {
            "tools": [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.name,
                            description=t.description,
                            parameters_json_schema=t.input_schema,
                        )
                        for t in tools
                    ]
                )
            ],
            # O SDK não executa funções sozinho: o agente controla e audita cada chamada.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }

    def next_turn(self) -> AssistantTurn:
        response = self._provider.generate(
            self._purpose, self._system, self._contents, self._config
        )
        self._contents.append(response.candidates[0].content)
        calls = [
            ToolCall(id=fc.id or uuid.uuid4().hex, name=fc.name, arguments=dict(fc.args or {}))
            for fc in response.function_calls or []
        ]
        text = "".join(
            p.text for p in response.candidates[0].content.parts or [] if p.text and not p.thought
        )
        return AssistantTurn(text=text, tool_calls=calls)

    def add_tool_results(self, outcomes: list[ToolOutcome]) -> None:
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=o.call.id,
                    name=o.call.name,
                    response={"error" if o.is_error else "output": json.loads(o.content)},
                )
            )
            for o in outcomes
        ]
        self._contents.append(types.Content(role="user", parts=parts))
