"""Montagem do relatório final (Markdown + HTML).

Números, tabelas, gráficos e lista de fontes vêm dos dados determinísticos coletados pelas
tools; o LLM contribui apenas com os textos interpretativos (`ReportAnalysis`).
"""

from datetime import UTC, datetime
from pathlib import Path

import markdown

from srag_agent.analysis import ReportAnalysis
from srag_agent.metrics import format_percent
from srag_agent.tools import RunContext

DISCLAIMER = (
    "Relatório gerado automaticamente por um agente de IA Generativa a partir de dados "
    "públicos do SIVEP-Gripe (Open DATASUS) e de notícias. Os números foram calculados de forma "
    "determinística; os textos interpretativos foram produzidos por um LLM e validados por "
    "regras automáticas. Use como apoio à vigilância, não como orientação clínica individual."
)

HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #0b0b0b;
         background: #f9f9f7; max-width: 960px; margin: 0 auto; padding: 24px 16px;
         line-height: 1.55; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #e4e3df; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .95rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e4e3df; }}
  th {{ color: #52514e; font-weight: 600; }}
  td:nth-child(2) {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #e4e3df; border-radius: 6px; }}
  blockquote {{ margin: 1rem 0; padding: .6rem 1rem; background: #fff7e6;
               border-left: 4px solid #eda100; }}
  .meta, small {{ color: #52514e; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _metrics_table(ctx: RunContext) -> str:
    snapshot = ctx.metrics[ctx.scope]
    rows = [
        "| Métrica | Valor | Numerador / denominador | Período (início dos sintomas) |",
        "|---|---|---|---|",
    ]
    for m in snapshot.metrics:
        rows.append(
            f"| {m.name} | {format_percent(m.value)} | "
            f"{_int(m.numerator)} / {_int(m.denominator)} | "
            f"{m.period_start:%d/%m/%Y} a {m.period_end:%d/%m/%Y} |"
        )
    return "\n".join(rows)


def _metric_sections(ctx: RunContext, analysis: ReportAnalysis) -> str:
    comments = {c.chave: c.comentario for c in analysis.analise_metricas}
    sections = []
    for m in ctx.metrics[ctx.scope].metrics:
        caveats = "\n".join(f"- {c}" for c in m.caveats)
        sections.append(
            f"### {m.name}: {format_percent(m.value)}\n\n"
            f"{comments.get(m.key, '_Sem comentário gerado._')}\n\n"
            f"<small>**Como é calculada:** {m.definition}</small>\n\n"
            f"<small>**Limitações:**</small>\n\n{caveats}\n"
        )
    return "\n".join(sections)


def _news_section(ctx: RunContext, analysis: ReportAnalysis) -> str:
    screening = ctx.news.get(ctx.scope)
    if not screening or not screening.accepted:
        return "_Nenhuma notícia disponível no momento da geração._"
    lines = [analysis.contexto_noticias, "", "**Notícias consultadas:**", ""]
    cited = set(analysis.noticias_citadas)
    for i, article in enumerate(screening.accepted, start=1):
        date = f"{article.published_at:%d/%m/%Y}" if article.published_at else "s/d"
        mark = " ✱" if i in cited else ""
        lines.append(f"{i}. [{article.title}]({article.url}) — {article.source}, {date}{mark}")
    lines.append("")
    lines.append("<small>✱ notícia usada na análise.</small>")
    if screening.rejected:
        lines.append(
            f"\n<small>{len(screening.rejected)} notícia(s) descartada(s) pelos guardrails "
            "por conter possíveis instruções ao modelo.</small>"
        )
    return "\n".join(lines)


def build_markdown(ctx: RunContext, analysis: ReportAnalysis, warnings: list[str]) -> str:
    charts = ctx.charts[ctx.scope]
    period = ctx.period
    generated = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")
    warning_block = "\n".join(f"> ⚠️ {w}" for w in warnings)
    attention = "\n".join(f"- {p}" for p in analysis.pontos_de_atencao)

    return f"""# Relatório de SRAG — {ctx.scope}

<p class="meta">Gerado em {generated} · Base atualizada até {period.data_cutoff:%d/%m/%Y} ·
Métricas até {period.reference_date:%d/%m/%Y} · Execução <code>{ctx.audit.run_id}</code></p>

{warning_block}

## Resumo executivo

{analysis.resumo_executivo}

## Métricas

{_metrics_table(ctx)}

<small>As métricas usam dados até {period.reference_date:%d/%m/%Y}: os {period.lag_days} dias
mais recentes da base ainda estão sendo digitados no SIVEP-Gripe e foram excluídos para não
confundir atraso de notificação com queda de casos.</small>

{_metric_sections(ctx, analysis)}

## Evolução dos casos

![Casos diários dos últimos 30 dias]({charts.daily_path.name})

![Casos mensais dos últimos 12 meses]({charts.monthly_path.name})

{analysis.tendencia_casos}

## Contexto: notícias recentes

{_news_section(ctx, analysis)}

## Pontos de atenção para a vigilância

{attention}

## Metodologia e governança

- **Fonte dos dados:** SIVEP-Gripe, dataset "SRAG 2019 a 2026" do Open DATASUS, tratado e
  anonimizado (sem identificadores, idade em faixas, localização por UF).
- **Cálculo:** consultas SQL fixas, em banco somente leitura; o LLM não calcula números.
- **Modelo:** `{ctx.model_label}` (orquestração das tools e redação da análise).
- **Validação:** percentuais citados conferidos contra os valores calculados, citações de
  notícias verificadas e mascaramento de dados pessoais.
- **Auditoria:** todas as decisões desta execução estão em
  `logs/audit/{ctx.audit.run_id}.jsonl`.

---

<small>{DISCLAIMER}</small>
"""


def render_report(ctx: RunContext, analysis: ReportAnalysis, warnings: list[str]) -> Path:
    """Grava `relatorio.md` e `relatorio.html` na pasta da execução e retorna o caminho do HTML."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    content = build_markdown(ctx, analysis, warnings)
    md_path = ctx.output_dir / "relatorio.md"
    md_path.write_text(content, encoding="utf-8")

    body = markdown.markdown(content, extensions=["tables"])
    html_path = ctx.output_dir / "relatorio.html"
    html_path.write_text(
        HTML_TEMPLATE.format(title=f"Relatório de SRAG — {ctx.scope}", body=body),
        encoding="utf-8",
    )
    return html_path
