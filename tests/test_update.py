"""Descoberta da publicação mais recente e atualização da base (sem rede)."""

from datetime import date

import httpx
import pytest

from srag_agent.data import sources, update
from srag_agent.data.sources import (
    SourceDiscoveryError,
    find_sources,
    latest_per_year,
    parse_source,
    resolve_sources,
)

PORTAL_HTML = """
<a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD24-23-03-2026.parquet">
<a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-28-09-2026.parquet">
<a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2026/INFLUD26-28-09-2026.parquet">
<a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2026/INFLUD26-12-10-2026.parquet">
<a href="https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2026/INFLUD26-12-10-2026.csv">
"""


def test_parse_source_reads_year_and_publication_date():
    source = parse_source("2026/INFLUD26-12-10-2026.parquet")
    assert (source.year, source.published) == (2026, date(2026, 10, 12))
    assert source.remote_path == "2026/INFLUD26-12-10-2026.parquet"
    assert parse_source("dicionario-de-dados.pdf") is None
    assert parse_source("INFLUD26-31-02-2026.parquet") is None  # data impossível


def test_latest_publication_per_year_is_selected():
    latest = latest_per_year(find_sources(PORTAL_HTML), [2025, 2026])
    assert [s.name for s in latest] == [
        "INFLUD25-28-09-2026.parquet",
        "INFLUD26-12-10-2026.parquet",  # a mais nova de 2026, não a de 28/09
    ]


def test_missing_year_is_skipped_but_empty_result_fails():
    assert [s.year for s in latest_per_year(find_sources(PORTAL_HTML), [2026, 2027])] == [2026]
    with pytest.raises(SourceDiscoveryError):
        latest_per_year(find_sources(PORTAL_HTML), [2030])


def test_falls_back_to_configured_files_when_portal_is_down(monkeypatch):
    def offline(*args, **kwargs):
        raise httpx.ConnectError("sem rede")

    monkeypatch.setattr(sources.httpx, "get", offline)
    chosen, origin = resolve_sources(["2026/INFLUD26-28-09-2026.parquet"], date(2026, 10, 8))
    assert origin == "configuração"
    assert [s.name for s in chosen] == ["INFLUD26-28-09-2026.parquet"]


def test_update_skips_when_database_is_current(settings, monkeypatch):
    latest = latest_per_year(find_sources(PORTAL_HTML), [2025, 2026])
    monkeypatch.setattr(update, "resolve_sources", lambda *a: (latest, "portal"))
    monkeypatch.setattr(update, "current_source_names", lambda _: {s.name for s in latest})
    monkeypatch.setattr(update, "download_all", lambda *a, **k: pytest.fail("não deveria baixar"))

    result = update.update(settings=settings)
    assert result.updated is False


def test_update_downloads_runs_etl_and_removes_superseded(settings, tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in ["INFLUD26-28-09-2026.parquet", "INFLUD24-23-03-2026.parquet"]:
        (raw_dir / name).write_bytes(b"")
    settings = settings.model_copy(update={"raw_dir": raw_dir})

    latest = latest_per_year(find_sources(PORTAL_HTML), [2025, 2026])
    calls = {}
    monkeypatch.setattr(update, "resolve_sources", lambda *a: (latest, "portal"))
    monkeypatch.setattr(update, "current_source_names", lambda _: {"INFLUD26-28-09-2026.parquet"})
    monkeypatch.setattr(update, "download_all", lambda srcs: calls.setdefault("download", srcs))
    monkeypatch.setattr(
        update,
        "build_database",
        lambda db, files, sources_origin: calls.setdefault("etl", (files, sources_origin)),
    )

    result = update.update(settings=settings)

    assert result.updated is True
    assert calls["etl"] == (
        ["2025/INFLUD25-28-09-2026.parquet", "2026/INFLUD26-12-10-2026.parquet"],
        "portal",
    )
    # A versão antiga de 2026 sai; o arquivo de 2024 (outro ano) é preservado.
    assert result.removed_files == ["INFLUD26-28-09-2026.parquet"]
    assert (raw_dir / "INFLUD24-23-03-2026.parquet").exists()
