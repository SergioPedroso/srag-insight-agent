"""Configuração central do projeto, lida de variáveis de ambiente (prefixo SRAG_) e do .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPENDATASUS_BASE_URL = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SRAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
