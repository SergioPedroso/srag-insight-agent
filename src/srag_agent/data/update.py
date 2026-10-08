"""Atualização da base a partir da publicação mais recente do Open DATASUS.

Descobre a publicação mais nova de cada ano, baixa, roda o ETL e limpa versões antigas.

Uso:
    python -m srag_agent.data.update            # só atualiza se houver publicação nova
    python -m srag_agent.data.update --forcar   # refaz o ETL mesmo sem novidade

Se a página do portal estiver indisponível, usa os arquivos de `SRAG_SOURCE_FILES`.
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from srag_agent.config import Settings, get_settings
from srag_agent.data.download import download_all
from srag_agent.data.etl import META_SOURCE_FILES, METADATA_TABLE, build_database
from srag_agent.data.sources import SourceFile, local_sources, resolve_sources

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    updated: bool
    sources: list[SourceFile]
    origin: str
    removed_files: list[str]


def current_source_names(db_path: Path) -> set[str]:
    """Arquivos usados no último ETL (vazio se o banco ainda não existe)."""
    if not db_path.exists():
        return set()
    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute(
            f"SELECT valor FROM {METADATA_TABLE} WHERE chave = ?", [META_SOURCE_FILES]
        ).fetchone()
    return {name.strip() for name in row[0].split(",")} if row and row[0] else set()


def remove_superseded(raw_dir: Path, keep: list[SourceFile]) -> list[str]:
    """Apaga versões antigas dos mesmos anos (são cópias locais, baixáveis de novo).

    Arquivos de outros anos, baixados manualmente, são preservados.
    """
    keep_names = {s.name for s in keep}
    keep_years = {s.year for s in keep}
    removed = []
    for source in local_sources(raw_dir):
        if source.year in keep_years and source.name not in keep_names:
            (raw_dir / source.name).unlink(missing_ok=True)
            removed.append(source.name)
    return removed


def update(*, force: bool = False, settings: Settings | None = None) -> UpdateResult:
    settings = settings or get_settings()
    sources, origin = resolve_sources(settings.source_files, date.today())
    names = {s.name for s in sources}
    logger.info("Publicações selecionadas (%s): %s", origin, ", ".join(sorted(names)))

    if not force and names == current_source_names(settings.db_path):
        logger.info("A base já usa as publicações mais recentes; nada a fazer.")
        return UpdateResult(updated=False, sources=sources, origin=origin, removed_files=[])

    download_all(sources)
    build_database(settings.db_path, [s.remote_path for s in sources], sources_origin=origin)
    removed = remove_superseded(settings.raw_dir, sources)
    if removed:
        logger.info("Publicações antigas removidas de %s: %s", settings.raw_dir, removed)
    return UpdateResult(updated=True, sources=sources, origin=origin, removed_files=removed)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Atualiza a base de SRAG a partir do portal.")
    parser.add_argument("--forcar", action="store_true", help="refaz o ETL mesmo sem novidade")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    update(force=args.forcar)


if __name__ == "__main__":
    main()
