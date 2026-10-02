"""Configuração central do projeto, lida de variáveis de ambiente (prefixo SRAG_) e do .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPENDATASUS_BASE_URL = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG"

DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "claude": "claude-opus-5-5",
}
# Modelo reserva quando o principal está sobrecarregado (a Claude faz isso no servidor).
DEFAULT_FALLBACK_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "claude": None,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SRAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Provedor do LLM: "gemini" (plano gratuito do Google AI Studio) ou "claude".
    llm_provider: Literal["gemini", "claude"] = "gemini"
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    # Vazio = modelo padrão do provedor (DEFAULT_MODELS).
    llm_model: str | None = None
    llm_fallback_model: str | None = None
    # Profundidade de raciocínio na Claude (low | medium | high | xhigh | max).
    llm_effort: str = "medium"
    llm_max_tokens: int = 16000
    # Limite de iterações do orquestrador (guardrail contra loops de tool use).
    max_agent_steps: int = 8

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    db_path: Path = PROJECT_ROOT / "data" / "processed" / "srag.duckdb"
    logs_dir: Path = PROJECT_ROOT / "logs"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    # Arquivos publicados no Open DATASUS (dataset "SRAG 2019 a 2026").
    # Os nomes mudam a cada atualização do portal; ajuste aqui ou via SRAG_SOURCE_FILES.
    source_files: list[str] = [
        "2025/INFLUD25-28-09-2026.parquet",
        "2026/INFLUD26-28-09-2026.parquet",
    ]
    data_dictionary_file: str = "dicionario-de-dados-2019-a-2025.pdf"

    # Casos recentes demoram a ser digitados no SIVEP-Gripe. As métricas usam uma
    # data de referência deslocada por este atraso para não confundir atraso de
    # notificação com queda real de casos.
    reporting_lag_days: int = 14

    @property
    def model_name(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    @property
    def fallback_model_name(self) -> str | None:
        return self.llm_fallback_model or DEFAULT_FALLBACK_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
