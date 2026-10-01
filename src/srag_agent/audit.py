"""Trilha de auditoria: cada execução grava um JSONL com as decisões do agente.

Registra, em ordem: requisição, nós do grafo, chamadas a tools (argumentos e resumo do
resultado), chamadas ao LLM (modelo, tokens, hash do prompt), decisões de guardrails e o
relatório final. Conteúdo grande é resumido por hash SHA-256 para manter o log enxuto e
ainda assim permitir verificar a integridade do que foi usado.
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def sha256(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    return value if isinstance(value, str | int | float | bool | type(None)) else str(value)


class AuditTrail:
    """Log append-only de uma execução. Uma instância por relatório."""

    def __init__(self, logs_dir: Path, run_id: str | None = None):
        self.run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.path = logs_dir / "audit" / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    def record(self, event: str, actor: str, **payload: Any) -> None:
        self._sequence += 1
        entry = {
            "run_id": self.run_id,
            "seq": self._sequence,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "actor": actor,
            **_to_jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
