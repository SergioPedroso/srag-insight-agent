"""Acesso ao banco analítico. O agente só enxerga o banco em modo somente leitura."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from srag_agent.config import get_settings


@contextmanager
def connect_readonly(db_path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Abre o DuckDB em modo somente leitura: qualquer escrita levanta erro no próprio banco."""
    path = db_path or get_settings().db_path
    if not path.exists():
        raise FileNotFoundError(
            f"Banco não encontrado em {path}. Rode antes: python -m srag_agent.data.etl"
        )
    con = duckdb.connect(str(path), read_only=True)
    try:
        yield con
    finally:
        con.close()
