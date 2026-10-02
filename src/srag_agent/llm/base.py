"""Interface comum aos provedores de LLM.

O agente conversa apenas com esta interface. Cada provedor (Gemini, Claude) mantém sua
própria transcrição da conversa de tool use, para preservar metadados que a API exige de
volta (por exemplo, blocos de raciocínio e assinaturas), e registra cada chamada na
auditoria no mesmo formato.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from srag_agent.audit import AuditTrail, sha256

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMRefusalError(RuntimeError):
    """O modelo recusou a requisição (política de segurança do provedor)."""


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolOutcome:
    call: ToolCall
    content: str  # JSON
    is_error: bool


class ToolSession(ABC):
    """Conversa de tool use: o modelo pede tools, o agente devolve os resultados."""

    @abstractmethod
    def next_turn(self) -> AssistantTurn: ...

    @abstractmethod
    def add_tool_results(self, outcomes: list[ToolOutcome]) -> None: ...


class LLMProvider(ABC):
    provider_name: str

    def __init__(self, model: str, audit: AuditTrail):
        self.model = model
        self._audit = audit

    @abstractmethod
    def start_tool_session(
        self, purpose: str, system: str, user_message: str, tools: list[ToolDefinition]
    ) -> ToolSession: ...

    @abstractmethod
    def structured(
        self, purpose: str, system: str, user_message: str, output_format: type[ModelT]
    ) -> ModelT: ...

    def record_call(
        self,
        purpose: str,
        *,
        system: str,
        payload: Any,
        served_model: str | None,
        stop_reason: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        request_id: str | None = None,
    ) -> None:
        self._audit.record(
            "llm_call",
            purpose,
            provider=self.provider_name,
            requested_model=self.model,
            served_model=served_model,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            system_prompt_sha256=sha256(system),
            payload_sha256=sha256(json.dumps(payload, default=str, ensure_ascii=False)),
            request_id=request_id,
        )
