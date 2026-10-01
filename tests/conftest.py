"""Fixtures compartilhadas: uma base sintética com contagens conhecidas."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from srag_agent.config import Settings
from srag_agent.data.etl import CASES_TABLE

DATA_CUTOFF = date(2026, 9, 28)  # referência = 14/09/2026 com atraso de 14 dias


def _case(day, uf="SP", *, uti=None, evolucao=None, vac=None, flu=None, hosp=True):
    return {
        "dt_sintomas": day,
        "dt_digitacao": DATA_CUTOFF,
        "uf_notificacao": uf,
        "hospitalizado": hosp,
        "internado_uti": uti,
        "evolucao": evolucao,
        "vacinado_covid": vac,
        "vacinado_gripe": flu,
    }


def synthetic_cases() -> list[dict]:
    cases = []
    # Semana anterior (01-07/09): 20 casos em SP.
    cases += [_case(date(2026, 9, 3)) for _ in range(20)]
    # Semana atual (08-14/09): 20 em SP + 10 no RJ.
    current = [_case(date(2026, 9, 10)) for _ in range(20)]
    current += [_case(date(2026, 9, 10), "RJ") for _ in range(10)]
    for i, case in enumerate(current):
        case["internado_uti"] = True if i < 6 else (False if i < 24 else None)
        case["vacinado_covid"] = True if i < 12 else (False if i < 24 else None)
        case["vacinado_gripe"] = i < 9
    cases += current
    # Agosto (janela de mortalidade): 40 casos em SP com desfechos conhecidos em 36.
    outcomes = ["Óbito por SRAG"] * 4 + ["Cura"] * 30 + ["Óbito por outras causas"] * 2
    outcomes += [None] * 4
    cases += [_case(date(2026, 8, 1), evolucao=o) for o in outcomes]
    # Caso recente, ainda no período sujeito a atraso de digitação.
    cases.append(_case(date(2026, 9, 25)))
    return cases


@pytest.fixture
def synthetic_db(tmp_path):
    db_path = tmp_path / "srag.duckdb"
    frame = pd.DataFrame(synthetic_cases())
    with duckdb.connect(str(db_path)) as con:
        con.register("frame", frame)
        con.execute(f"CREATE TABLE {CASES_TABLE} AS SELECT * FROM frame")
    return db_path


@pytest.fixture
def settings(tmp_path, synthetic_db):
    return Settings(
        db_path=synthetic_db,
        logs_dir=tmp_path / "logs",
        outputs_dir=tmp_path / "outputs",
        anthropic_api_key="test-key",
    )
