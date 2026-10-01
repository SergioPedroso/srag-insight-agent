"""Gráficos do relatório (PNG estático via matplotlib).

Uma série por gráfico, em um único tom; o período ainda sujeito a atraso de digitação
aparece em tom claro e hachurado (cor + textura, para não depender só da cor).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # renderização sem interface gráfica

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from srag_agent.metrics import SeriesPoint  # noqa: E402

COMPLETE_COLOR = "#2a78d6"
INCOMPLETE_COLOR = "#86b6ef"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRID = "#e4e3df"

MONTHS_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    )


def _bar_chart(
    points: list[SeriesPoint],
    *,
    title: str,
    subtitle: str,
    bar_width: float,
    x_formatter,
    x_locator,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for point in points:
        ax.bar(
            point.period,
            point.cases,
            width=bar_width,
            color=INCOMPLETE_COLOR if point.incomplete else COMPLETE_COLOR,
            hatch="///" if point.incomplete else None,
            edgecolor=SURFACE,
            linewidth=1,
        )

    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_title(subtitle, loc="left", fontsize=9, color=TEXT_MUTED, pad=10)

    if any(p.incomplete for p in points):
        ax.legend(
            handles=[
                Patch(facecolor=COMPLETE_COLOR, label="Dados consolidados"),
                Patch(
                    facecolor=INCOMPLETE_COLOR,
                    hatch="///",
                    edgecolor=SURFACE,
                    label="Sujeito a atraso de digitação",
                ),
            ],
            loc="upper right",
            frameon=False,
            fontsize=8,
            labelcolor=TEXT_MUTED,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def plot_daily_cases(points: list[SeriesPoint], scope: str, output_path: Path) -> Path:
    return _bar_chart(
        points,
        title=f"Casos diários de SRAG — últimos 30 dias ({scope})",
        subtitle="Casos por data de início dos sintomas. Fonte: SIVEP-Gripe / Open DATASUS.",
        bar_width=0.8,
        x_locator=mdates.DayLocator(interval=3),
        x_formatter=mdates.DateFormatter("%d/%m"),
        output_path=output_path,
    )


def plot_monthly_cases(points: list[SeriesPoint], scope: str, output_path: Path) -> Path:
    return _bar_chart(
        points,
        title=f"Casos mensais de SRAG — últimos 12 meses ({scope})",
        subtitle="Casos por mês de início dos sintomas. Fonte: SIVEP-Gripe / Open DATASUS.",
        bar_width=24,
        x_locator=mdates.MonthLocator(),
        x_formatter=matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{MONTHS_PT[mdates.num2date(v).month - 1]}/{mdates.num2date(v):%y}"
        ),
        output_path=output_path,
    )
