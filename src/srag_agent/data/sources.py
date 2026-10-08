"""Descoberta das publicações mais recentes de SRAG no Open DATASUS.

O portal publica um Parquet por ano e troca o nome do arquivo a cada atualização
(`INFLUD26-28-09-2026.parquet` = dados de 2026 publicados em 28/09/2026). Este módulo lê a
página do dataset, interpreta esses nomes e escolhe a publicação mais recente de cada ano.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DATASET_PAGE_URL = "https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026"

_SOURCE_RE = re.compile(
    r"(?:(?P<folder>\d{4})/)?(?P<name>INFLUD(?P<yy>\d{2})-(?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})"
    r"\.parquet)"
)


class SourceDiscoveryError(RuntimeError):
    """Não foi possível identificar as publicações no portal."""


@dataclass(frozen=True)
class SourceFile:
    year: int  # ano dos casos
    published: date  # data da publicação no portal
    name: str

    @property
    def remote_path(self) -> str:
        return f"{self.year}/{self.name}"


def parse_source(text: str) -> SourceFile | None:
    """Interpreta um nome ou caminho como `2026/INFLUD26-28-09-2026.parquet`."""
    match = _SOURCE_RE.search(text)
    if not match:
        return None
    try:
        published = date(int(match["y"]), int(match["m"]), int(match["d"]))
    except ValueError:
        return None
    return SourceFile(year=2000 + int(match["yy"]), published=published, name=match["name"])


def find_sources(text: str) -> list[SourceFile]:
    """Todas as publicações citadas em um texto (por exemplo, o HTML da página do dataset)."""
    found = {source for m in _SOURCE_RE.finditer(text) if (source := parse_source(m.group(0)))}
    return sorted(found, key=lambda s: (s.year, s.published))


def required_years(today: date) -> list[int]:
    """Anos necessários para cobrir os últimos 12 meses (ano anterior e ano corrente)."""
    return [today.year - 1, today.year]


def latest_per_year(sources: list[SourceFile], years: list[int]) -> list[SourceFile]:
    """Publicação mais recente de cada ano pedido. Anos sem publicação são ignorados."""
    latest = []
    for year in years:
        candidates = [s for s in sources if s.year == year]
        if candidates:
            latest.append(max(candidates, key=lambda s: s.published))
        else:
            logger.warning("Nenhuma publicação encontrada para %s.", year)
    if not latest:
        raise SourceDiscoveryError(f"Nenhuma publicação encontrada para os anos {years}.")
    return latest


def discover_latest(years: list[int], *, timeout: float = 60) -> list[SourceFile]:
    """Lê a página do dataset e devolve a publicação mais recente de cada ano."""
    response = httpx.get(DATASET_PAGE_URL, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return latest_per_year(find_sources(response.text), years)


def local_sources(raw_dir: Path) -> list[SourceFile]:
    """Publicações já baixadas em `raw_dir`."""
    return find_sources("\n".join(p.name for p in raw_dir.glob("INFLUD*.parquet")))


def resolve_sources(configured: list[str], today: date) -> tuple[list[SourceFile], str]:
    """Publicações a usar: as mais recentes do portal ou, se a descoberta falhar, as configuradas.

    Retorna também a origem da decisão ("portal" ou "configuração"), registrada nos metadados.
    """
    try:
        return discover_latest(required_years(today)), "portal"
    except (httpx.HTTPError, SourceDiscoveryError) as exc:
        logger.warning("Descoberta no portal falhou (%s); usando os arquivos configurados.", exc)
    fallback = [s for name in configured if (s := parse_source(name))]
    if not fallback:
        raise SourceDiscoveryError("Nenhuma publicação válida em SRAG_SOURCE_FILES.")
    return fallback, "configuração"
