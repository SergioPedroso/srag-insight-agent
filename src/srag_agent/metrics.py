"""Cálculo determinístico das métricas e séries do relatório.

O LLM nunca calcula números: todas as métricas saem daqui, de consultas SQL fixas e
parametrizadas sobre a tabela anonimizada. Cada métrica carrega numerador, denominador,
período, definição e limitações, para que o relatório seja auditável.

Atraso de notificação: casos recentes ainda estão sendo digitados no SIVEP-Gripe.
As métricas usam uma *data de referência* = última data de digitação - `lag_days`;
nas séries, os dias/meses posteriores a ela são marcados como incompletos.
"""

from datetime import date, timedelta

import duckdb
from pydantic import BaseModel, Field

from srag_agent.data.etl import CASES_TABLE
from srag_agent.geo import BRAZILIAN_UFS

GROWTH_WINDOW_DAYS = 7
MORTALITY_WINDOW_DAYS = 90
ICU_WINDOW_DAYS = 30
VACCINATION_WINDOW_DAYS = 30


class ReportPeriod(BaseModel):
    data_cutoff: date = Field(description="Data de digitação mais recente na base")
    reference_date: date = Field(description="Último dia considerado completo para métricas")
    lag_days: int


class Metric(BaseModel):
    key: str
    name: str
    value: float | None = Field(description="Proporção (0.25 = 25%); None se denominador = 0")
    numerator: int
    denominator: int
    period_start: date
    period_end: date
    definition: str
    caveats: list[str] = []


class SeriesPoint(BaseModel):
    period: date
    cases: int
    incomplete: bool


class MetricsSnapshot(BaseModel):
    scope: str
    period: ReportPeriod
    metrics: list[Metric]


class InvalidScopeError(ValueError):
    pass


def normalize_uf(uf: str | None) -> str | None:
    """Valida o filtro geográfico contra a lista fechada de UFs (nunca vai cru para o SQL)."""
    if uf is None or not uf.strip() or uf.strip().upper() in {"BR", "BRASIL"}:
        return None
    normalized = uf.strip().upper()
    if normalized not in BRAZILIAN_UFS:
        raise InvalidScopeError(f"UF inválida: {uf!r}. Use uma sigla como 'SP' ou deixe vazio.")
    return normalized


def scope_label(uf: str | None) -> str:
    return uf or "Brasil"


def _uf_filter(uf: str | None) -> tuple[str, list]:
    return ("AND uf_notificacao = ?", [uf]) if uf else ("", [])


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def format_percent(value: float | None) -> str:
    """Proporção no formato brasileiro (0.123 -> '12,3%')."""
    return "indisponível" if value is None else f"{value * 100:.1f}%".replace(".", ",")


def get_report_period(con: duckdb.DuckDBPyConnection, lag_days: int) -> ReportPeriod:
    cutoff = con.execute(f"SELECT max(dt_digitacao) FROM {CASES_TABLE}").fetchone()[0]
    if cutoff is None:
        raise ValueError("A tabela de casos está vazia.")
    return ReportPeriod(
        data_cutoff=cutoff,
        reference_date=cutoff - timedelta(days=lag_days),
        lag_days=lag_days,
    )


def case_growth_rate(con, period: ReportPeriod, uf: str | None = None) -> Metric:
    end = period.reference_date
    current_start = end - timedelta(days=GROWTH_WINDOW_DAYS - 1)
    previous_start = current_start - timedelta(days=GROWTH_WINDOW_DAYS)
    uf_sql, uf_params = _uf_filter(uf)
    current, previous = con.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE dt_sintomas >= ?),
            count(*) FILTER (WHERE dt_sintomas < ?)
        FROM {CASES_TABLE}
        WHERE dt_sintomas BETWEEN ? AND ? {uf_sql}
        """,
        [current_start, current_start, previous_start, end, *uf_params],
    ).fetchone()
    growth = (current - previous) / previous if previous else None
    return Metric(
        key="taxa_aumento_casos",
        name="Taxa de aumento de casos",
        value=growth,
        numerator=current,
        denominator=previous,
        period_start=current_start,
        period_end=end,
        definition=(
            f"Variação percentual dos casos de SRAG (por data de início dos sintomas) nos "
            f"últimos {GROWTH_WINDOW_DAYS} dias completos em relação aos {GROWTH_WINDOW_DAYS} "
            "dias anteriores. Numerador = casos da semana atual; denominador = semana anterior."
        ),
        caveats=[
            f"Os últimos {period.lag_days} dias foram excluídos por atraso de digitação; "
            "mesmo assim, a semana mais recente pode ser revisada para cima.",
        ],
    )


def mortality_rate(con, period: ReportPeriod, uf: str | None = None) -> Metric:
    end = period.reference_date
    start = end - timedelta(days=MORTALITY_WINDOW_DAYS - 1)
    uf_sql, uf_params = _uf_filter(uf)
    deaths, closed, total = con.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE evolucao = 'Óbito por SRAG'),
            count(*) FILTER (WHERE evolucao IS NOT NULL),
            count(*)
        FROM {CASES_TABLE}
        WHERE dt_sintomas BETWEEN ? AND ? {uf_sql}
        """,
        [start, end, *uf_params],
    ).fetchone()
    open_share = _ratio(total - closed, total) or 0.0
    return Metric(
        key="taxa_mortalidade",
        name="Taxa de mortalidade (letalidade hospitalar)",
        value=_ratio(deaths, closed),
        numerator=deaths,
        denominator=closed,
        period_start=start,
        period_end=end,
        definition=(
            "Óbitos por SRAG divididos pelos casos com desfecho conhecido (cura, óbito por "
            f"SRAG ou óbito por outras causas), entre casos com sintomas nos últimos "
            f"{MORTALITY_WINDOW_DAYS} dias completos."
        ),
        caveats=[
            "Mede letalidade entre casos notificados de SRAG (majoritariamente hospitalizados), "
            "não a mortalidade da população geral.",
            f"{format_percent(open_share)} dos casos do período ainda não têm desfecho registrado.",
        ],
    )


def icu_rate(con, period: ReportPeriod, uf: str | None = None) -> Metric:
    end = period.reference_date
    start = end - timedelta(days=ICU_WINDOW_DAYS - 1)
    uf_sql, uf_params = _uf_filter(uf)
    icu, known = con.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE internado_uti),
            count(*) FILTER (WHERE internado_uti IS NOT NULL)
        FROM {CASES_TABLE}
        WHERE hospitalizado AND dt_sintomas BETWEEN ? AND ? {uf_sql}
        """,
        [start, end, *uf_params],
    ).fetchone()
    return Metric(
        key="taxa_ocupacao_uti",
        name="Taxa de ocupação de UTI (proxy)",
        value=_ratio(icu, known),
        numerator=icu,
        denominator=known,
        period_start=start,
        period_end=end,
        definition=(
            "Proporção dos casos de SRAG hospitalizados que foram internados em UTI, entre "
            f"casos com sintomas nos últimos {ICU_WINDOW_DAYS} dias completos e com a "
            "informação de UTI preenchida."
        ),
        caveats=[
            "A base não informa a capacidade de leitos; por isso esta é uma proxy da pressão "
            "sobre UTIs, não a ocupação de leitos propriamente dita.",
        ],
    )


def vaccination_rate(con, period: ReportPeriod, uf: str | None = None) -> Metric:
    end = period.reference_date
    start = end - timedelta(days=VACCINATION_WINDOW_DAYS - 1)
    uf_sql, uf_params = _uf_filter(uf)
    vaccinated, known, flu_vaccinated, flu_known = con.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE vacinado_covid),
            count(*) FILTER (WHERE vacinado_covid IS NOT NULL),
            count(*) FILTER (WHERE vacinado_gripe),
            count(*) FILTER (WHERE vacinado_gripe IS NOT NULL)
        FROM {CASES_TABLE}
        WHERE dt_sintomas BETWEEN ? AND ? {uf_sql}
        """,
        [start, end, *uf_params],
    ).fetchone()
    flu_rate = _ratio(flu_vaccinated, flu_known)
    flu_text = format_percent(flu_rate)
    return Metric(
        key="taxa_vacinacao",
        name="Taxa de vacinação (COVID-19) entre casos de SRAG",
        value=_ratio(vaccinated, known),
        numerator=vaccinated,
        denominator=known,
        period_start=start,
        period_end=end,
        definition=(
            "Proporção dos casos de SRAG que receberam ao menos uma dose de vacina contra "
            f"COVID-19, entre casos com sintomas nos últimos {VACCINATION_WINDOW_DAYS} dias "
            "completos e com a informação preenchida."
        ),
        caveats=[
            "A base contém apenas pessoas que tiveram SRAG; a métrica não representa a "
            "cobertura vacinal da população geral.",
            f"Vacinação contra gripe na última campanha entre os mesmos casos: {flu_text} "
            f"({flu_vaccinated} de {flu_known}).",
        ],
    )


def compute_metrics(
    con: duckdb.DuckDBPyConnection, period: ReportPeriod, uf: str | None = None
) -> MetricsSnapshot:
    uf = normalize_uf(uf)
    return MetricsSnapshot(
        scope=scope_label(uf),
        period=period,
        metrics=[
            case_growth_rate(con, period, uf),
            mortality_rate(con, period, uf),
            icu_rate(con, period, uf),
            vaccination_rate(con, period, uf),
        ],
    )


def daily_cases(
    con: duckdb.DuckDBPyConnection, period: ReportPeriod, uf: str | None = None, days: int = 30
) -> list[SeriesPoint]:
    """Casos por dia de início dos sintomas nos últimos `days` dias até a data de corte."""
    uf = normalize_uf(uf)
    end = period.data_cutoff
    start = end - timedelta(days=days - 1)
    uf_sql, uf_params = _uf_filter(uf)
    rows = con.execute(
        f"""
        WITH calendar AS (
            SELECT CAST(d AS DATE) AS day FROM range(?::DATE, ?::DATE + 1, INTERVAL 1 DAY) t(d)
        ), counts AS (
            SELECT dt_sintomas AS day, count(*) AS cases
            FROM {CASES_TABLE}
            WHERE dt_sintomas BETWEEN ? AND ? {uf_sql}
            GROUP BY 1
        )
        SELECT calendar.day, coalesce(counts.cases, 0)
        FROM calendar LEFT JOIN counts USING (day)
        ORDER BY 1
        """,
        [start, end, start, end, *uf_params],
    ).fetchall()
    return [
        SeriesPoint(period=day, cases=cases, incomplete=day > period.reference_date)
        for day, cases in rows
    ]


def monthly_cases(
    con: duckdb.DuckDBPyConnection, period: ReportPeriod, uf: str | None = None, months: int = 12
) -> list[SeriesPoint]:
    """Casos por mês de início dos sintomas nos últimos `months` meses (inclui o mês corrente)."""
    uf = normalize_uf(uf)
    last_month = period.data_cutoff.replace(day=1)
    first_month = _add_months(last_month, -(months - 1))
    uf_sql, uf_params = _uf_filter(uf)
    rows = con.execute(
        f"""
        WITH calendar AS (
            SELECT CAST(m AS DATE) AS month
            FROM range(?::DATE, ?::DATE + INTERVAL 1 MONTH, INTERVAL 1 MONTH) t(m)
        ), counts AS (
            SELECT CAST(date_trunc('month', dt_sintomas) AS DATE) AS month, count(*) AS cases
            FROM {CASES_TABLE}
            WHERE dt_sintomas BETWEEN ? AND ? {uf_sql}
            GROUP BY 1
        )
        SELECT calendar.month, coalesce(counts.cases, 0)
        FROM calendar LEFT JOIN counts USING (month)
        ORDER BY 1
        """,
        [first_month, last_month, first_month, period.data_cutoff, *uf_params],
    ).fetchall()
    return [
        SeriesPoint(
            period=month,
            cases=cases,
            incomplete=_add_months(month, 1) - timedelta(days=1) > period.reference_date,
        )
        for month, cases in rows
    ]


def _add_months(day: date, months: int) -> date:
    month_index = day.year * 12 + day.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)
