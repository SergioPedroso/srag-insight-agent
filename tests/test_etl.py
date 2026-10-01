from datetime import datetime

import duckdb
import pandas as pd
import pytest

from srag_agent.config import Settings
from srag_agent.data import etl

SENSITIVE_RAW_COLUMNS = {"NU_NOTIFIC", "CO_MUN_NOT", "DT_NASC", "NU_IDADE_N", "NM_UN_INTE"}


def _raw_row(**overrides):
    row = {
        "NU_NOTIFIC": "123",
        "CO_MUN_NOT": "355030",
        "DT_NOTIFIC": datetime(2026, 3, 5),
        "DT_SIN_PRI": datetime(2026, 3, 1),
        "DT_DIGITA": datetime(2026, 3, 10),
        "DT_INTERNA": datetime(2026, 3, 4),
        "DT_ENTUTI": None,
        "DT_SAIDUTI": None,
        "DT_EVOLUCA": datetime(2026, 3, 9),
        "SG_UF_NOT": "SP",
        "SG_UF": "SP",
        "CS_SEXO": "F",
        "NU_IDADE_N": 34,
        "TP_IDADE": "3",
        "HOSPITAL": "1",
        "UTI": "2",
        "SUPORT_VEN": "3",
        "CLASSI_FIN": 5,
        "EVOLUCAO": "1",
        "VACINA_COV": "1",
        "VACINA": "9",
        # Colunas sensíveis presentes no bruto que não podem chegar ao banco.
        "DT_NASC": datetime(1991, 7, 2),
        "NM_UN_INTE": "HOSPITAL EXEMPLO",
    }
    row.update(overrides)
    return row


@pytest.fixture
def built_db(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _raw_row(),
        _raw_row(NU_NOTIFIC="124", NU_IDADE_N=8, TP_IDADE="2", UTI="1", EVOLUCAO="2"),
        # Sintomas depois da notificação: deve ser descartado.
        _raw_row(NU_NOTIFIC="125", DT_SIN_PRI=datetime(2026, 3, 8)),
        # Data de evolução anterior aos sintomas: mantém o caso, anula a data.
        _raw_row(NU_NOTIFIC="126", DT_EVOLUCA=datetime(2026, 2, 1)),
    ]
    pd.DataFrame(rows).to_parquet(raw_dir / "INFLUD26-test.parquet")

    settings = Settings(raw_dir=raw_dir, source_files=["2026/INFLUD26-test.parquet"])
    monkeypatch.setattr(etl, "get_settings", lambda: settings)
    db_path = tmp_path / "srag.duckdb"
    quality = etl.build_database(db_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        cases = con.execute(f"SELECT * FROM {etl.CASES_TABLE} ORDER BY case_id").df()
    return quality, cases


def test_sensitive_columns_never_reach_the_database(built_db):
    _, cases = built_db
    assert SENSITIVE_RAW_COLUMNS.isdisjoint(c.upper() for c in cases.columns)


def test_exact_age_is_replaced_by_age_band(built_db):
    _, cases = built_db
    assert set(cases["faixa_etaria"]) == {"30-39", "0-1"}


def test_invalid_dates_are_removed_or_nulled(built_db):
    quality, cases = built_db
    assert quality["datas_invalidas_removidas"] == 1
    assert len(cases) == 3
    assert cases["dt_evolucao"].isna().sum() == 1


def test_codes_are_decoded(built_db):
    _, cases = built_db
    first = cases.iloc[0]
    assert first["classificacao_final"] == "COVID-19"
    assert first["evolucao"] == "Cura"
    assert bool(first["vacinado_covid"]) is True
    assert pd.isna(first["vacinado_gripe"])  # 9-Ignorado vira NULL
