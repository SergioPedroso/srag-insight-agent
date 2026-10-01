from datetime import date

import duckdb
import pytest

from srag_agent.metrics import (
    InvalidScopeError,
    compute_metrics,
    daily_cases,
    get_report_period,
    monthly_cases,
    normalize_uf,
)


@pytest.fixture
def con(synthetic_db):
    with duckdb.connect(str(synthetic_db), read_only=True) as connection:
        yield connection


@pytest.fixture
def period(con):
    return get_report_period(con, lag_days=14)


def _by_key(snapshot):
    return {m.key: m for m in snapshot.metrics}


def test_reference_date_discounts_reporting_lag(period):
    assert period.data_cutoff == date(2026, 9, 28)
    assert period.reference_date == date(2026, 9, 14)


def test_brazil_metrics(con, period):
    metrics = _by_key(compute_metrics(con, period))

    growth = metrics["taxa_aumento_casos"]
    assert (growth.numerator, growth.denominator) == (30, 20)
    assert growth.value == pytest.approx(0.5)

    mortality = metrics["taxa_mortalidade"]
    assert (mortality.numerator, mortality.denominator) == (4, 36)

    icu = metrics["taxa_ocupacao_uti"]
    assert (icu.numerator, icu.denominator) == (6, 24)

    vaccination = metrics["taxa_vacinacao"]
    assert (vaccination.numerator, vaccination.denominator) == (12, 24)
    assert "30,0% (9 de 30)" in vaccination.caveats[-1]  # vacina contra gripe


def test_uf_filter(con, period):
    metrics = _by_key(compute_metrics(con, period, "sp"))
    assert metrics["taxa_aumento_casos"].value == pytest.approx(0.0)


def test_case_after_reference_date_is_not_counted(con, period):
    metrics = _by_key(compute_metrics(con, period))
    assert metrics["taxa_aumento_casos"].numerator == 30  # o caso de 25/09 fica de fora


def test_series_mark_incomplete_periods(con, period):
    daily = daily_cases(con, period)
    assert len(daily) == 30
    assert daily[-1].period == period.data_cutoff and daily[-1].incomplete
    assert next(p for p in daily if p.period == date(2026, 9, 10)).cases == 30

    monthly = monthly_cases(con, period)
    assert len(monthly) == 12
    assert monthly[-1].period == date(2026, 9, 1) and monthly[-1].incomplete
    assert next(p for p in monthly if p.period == date(2026, 8, 1)).cases == 40


@pytest.mark.parametrize("value", ["XX", "São Paulo", "SP; DROP TABLE srag_cases"])
def test_invalid_uf_is_rejected(value):
    with pytest.raises(InvalidScopeError):
        normalize_uf(value)


@pytest.mark.parametrize("value", [None, "", "BR", "brasil"])
def test_brazil_scope(value):
    assert normalize_uf(value) is None
