"""Agente orquestrador em LangGraph.

Fluxo (o pedido já chega validado pelos guardrails de entrada em `generate_report`):

    orquestrador <-> executar_tools
                            |
                     garantir_cobertura -> redigir_analise -> validar_saida -> montar_relatorio
                                                  ^                 |
                                                  +--- revisão -----+

- `orquestrador`: a Claude decide quais tools chamar (métricas, gráficos, notícias).
- `garantir_cobertura`: guardrail que executa deterministicamente qualquer tool obrigatória
  que o LLM não tenha chamado (ou se o limite de passos estourar).
- `redigir_analise`: a Claude escreve a análise em formato estruturado (Pydantic).
- `validar_saida`: confere números, citações de notícias e dados pessoais; reprovação gera
  uma nova tentativa com o feedback (até `MAX_WRITER_ATTEMPTS`).
"""

import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from srag_agent.analysis import ReportAnalysis
from srag_agent.audit import AuditTrail
from srag_agent.config import Settings, get_settings
from srag_agent.db import connect_readonly
from srag_agent.guardrails import (
    GuardrailViolation,
    allowed_percentages,
    contains_injection,
    find_unverified_counts,
    find_unverified_percentages,
    mask_pii,
    numbers_in,
    validate_request,
)
from srag_agent.llm import LLMProvider, ToolCall, ToolOutcome, ToolSession, create_provider
from srag_agent.metrics import get_report_period
from srag_agent.prompts import (
    FOCUS_BLOCK,
    ORCHESTRATOR_SYSTEM,
    REVISION_REQUEST,
    WRITER_SYSTEM,
    WRITER_USER_TEMPLATE,
)
from srag_agent.report import render_report
from srag_agent.tools import (
    REQUIRED_TOOLS,
    RunContext,
    execute_tool,
    metrics_payload,
    news_payload,
    series_payload,
    tool_definitions,
)

MAX_WRITER_ATTEMPTS = 2


class AgentState(TypedDict, total=False):
    pending_calls: list[ToolCall]
    steps: int
    analysis: ReportAnalysis
    writer_attempts: int
    validation_problems: list[str]
    warnings: list[str]
    report_path: Path


class SragReportAgent:
    def __init__(self, ctx: RunContext, llm: LLMProvider, settings: Settings):
        self.ctx = ctx
        self.llm = llm
        self.settings = settings
        self._session: ToolSession | None = None
        self.graph = self._build_graph()

    # ---- nós do grafo -------------------------------------------------------------------

    def orchestrate(self, state: AgentState) -> AgentState:
        if self._session is None:
            self._session = self.llm.start_tool_session(
                "orchestrator",
                ORCHESTRATOR_SYSTEM,
                f"Gere o relatório de SRAG para o escopo: "
                f"{self.ctx.request.uf or 'BR'} ({self.ctx.scope}).",
                tool_definitions(),
            )
        turn = self._session.next_turn()
        self.ctx.audit.record(
            "agent_decision",
            "orchestrator",
            tool_requests=[{"tool": c.name, "arguments": c.arguments} for c in turn.tool_calls],
            message=turn.text,
        )
        return {"pending_calls": turn.tool_calls, "steps": state.get("steps", 0) + 1}

    def run_tools(self, state: AgentState) -> AgentState:
        outcomes = []
        for call in state["pending_calls"]:
            content, is_error = execute_tool(self.ctx, call.name, call.arguments)
            outcomes.append(ToolOutcome(call=call, content=content, is_error=is_error))
        self._session.add_tool_results(outcomes)
        return {"pending_calls": []}

    def ensure_coverage(self, state: AgentState) -> AgentState:
        scope = self.ctx.scope
        collected = {
            "consultar_metricas_srag": scope in self.ctx.metrics,
            "gerar_graficos_casos": scope in self.ctx.charts,
            "buscar_noticias_srag": scope in self.ctx.news,
        }
        warnings = list(state.get("warnings", []))
        for tool in REQUIRED_TOOLS:
            if collected[tool]:
                continue
            self.ctx.audit.record(
                "guardrail",
                "coverage",
                decision="forced_tool_execution",
                tool=tool,
                reason="tool obrigatória não chamada pelo orquestrador",
            )
            _, is_error = execute_tool(self.ctx, tool, {"uf": self.ctx.request.uf or "BR"})
            if is_error:
                warnings.append(f"Não foi possível executar '{tool}'.")
        if scope not in self.ctx.metrics or scope not in self.ctx.charts:
            raise RuntimeError("Métricas ou gráficos indisponíveis; relatório não pode ser gerado.")
        if scope not in self.ctx.news:
            warnings.append("Notícias indisponíveis: análise sem contexto externo.")
        return {"warnings": warnings}

    def write_analysis(self, state: AgentState) -> AgentState:
        attempts = state.get("writer_attempts", 0) + 1
        user_content = self._writer_input()
        if state.get("validation_problems"):
            previous = state["analysis"].model_dump_json()
            user_content += "\n<versao_anterior>\n" + previous + "\n</versao_anterior>\n\n"
            user_content += REVISION_REQUEST.format(
                problems="\n".join(f"- {p}" for p in state["validation_problems"])
            )
        analysis = self.llm.structured(
            f"writer_attempt_{attempts}", WRITER_SYSTEM, user_content, ReportAnalysis
        )
        return {"analysis": analysis, "writer_attempts": attempts}

    def validate_output(self, state: AgentState) -> AgentState:
        analysis = state["analysis"]
        snapshot = self.ctx.metrics[self.ctx.scope]
        news = self.ctx.news.get(self.ctx.scope)
        articles = news.accepted if news else []
        text = analysis.full_text()

        problems = []
        unverified = find_unverified_percentages(
            text, allowed_percentages(snapshot, articles) | self._series_percentages()
        )
        if unverified:
            problems.append(
                "Percentuais sem correspondência nas métricas ou fontes: " + ", ".join(unverified)
            )
        wrong_counts = find_unverified_counts(text, self._allowed_counts(articles))
        if wrong_counts:
            problems.append(
                "Contagens de casos que não aparecem nos dados (copie os valores exatos das "
                "séries ou métricas): " + ", ".join(wrong_counts)
            )
        invalid_ids = [i for i in analysis.noticias_citadas if not 1 <= i <= len(articles)]
        if invalid_ids:
            problems.append(f"ids de notícias inexistentes citados: {invalid_ids}")
        if contains_injection(text):
            problems.append("O texto contém instruções ou trechos de prompt.")
        covered = {c.chave for c in analysis.analise_metricas}
        missing = {m.key for m in snapshot.metrics} - covered
        if missing:
            problems.append(f"Métricas sem comentário: {sorted(missing)}")

        _, pii = mask_pii(text)
        decision = "approved" if not problems else "rejected"
        self.ctx.audit.record(
            "guardrail",
            "output_validation",
            decision=decision,
            attempt=state["writer_attempts"],
            problems=problems,
            pii_masked=pii,
        )

        warnings = list(state.get("warnings", []))
        if problems and state["writer_attempts"] >= MAX_WRITER_ATTEMPTS:
            warnings.extend(f"Validação automática: {p}" for p in problems)
            problems = []  # segue com aviso explícito no relatório
        return {
            "analysis": analysis.masked(),
            "validation_problems": problems,
            "warnings": warnings,
        }

    def render(self, state: AgentState) -> AgentState:
        path = render_report(self.ctx, state["analysis"], state.get("warnings", []))
        self.ctx.audit.record("report_rendered", "renderer", path=path)
        return {"report_path": path}

    # ---- roteamento ---------------------------------------------------------------------

    def _after_orchestrator(self, state: AgentState) -> str:
        wants_tools = bool(state["pending_calls"])
        if wants_tools and state["steps"] < self.settings.max_agent_steps:
            return "executar_tools"
        if wants_tools:
            self.ctx.audit.record(
                "guardrail", "max_steps", decision="stop_loop", steps=state["steps"]
            )
        return "garantir_cobertura"

    def _after_validation(self, state: AgentState) -> str:
        return "redigir_analise" if state["validation_problems"] else "montar_relatorio"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("orquestrador", self.orchestrate)
        graph.add_node("executar_tools", self.run_tools)
        graph.add_node("garantir_cobertura", self.ensure_coverage)
        graph.add_node("redigir_analise", self.write_analysis)
        graph.add_node("validar_saida", self.validate_output)
        graph.add_node("montar_relatorio", self.render)

        graph.add_edge(START, "orquestrador")
        graph.add_conditional_edges("orquestrador", self._after_orchestrator)
        graph.add_edge("executar_tools", "orquestrador")
        graph.add_edge("garantir_cobertura", "redigir_analise")
        graph.add_edge("redigir_analise", "validar_saida")
        graph.add_conditional_edges("validar_saida", self._after_validation)
        graph.add_edge("montar_relatorio", END)
        return graph.compile()

    # ---- apoio --------------------------------------------------------------------------

    def _writer_input(self) -> str:
        scope = self.ctx.scope
        charts = self.ctx.charts[scope]
        series = {
            "casos_mensais": series_payload(charts.monthly, "%Y-%m"),
            "casos_diarios": series_payload(charts.daily, "%Y-%m-%d"),
        }
        focus = self.ctx.request.focus
        return WRITER_USER_TEMPLATE.format(
            scope=scope,
            data_cutoff=self.ctx.period.data_cutoff,
            reference_date=self.ctx.period.reference_date,
            focus_block=FOCUS_BLOCK.format(focus=focus) if focus else "",
            metrics_json=json.dumps(
                metrics_payload(self.ctx.metrics[scope]), ensure_ascii=False, default=str, indent=1
            ),
            series_json=json.dumps(series, ensure_ascii=False),
            news_json=json.dumps(
                news_payload(self.ctx.news.get(scope)), ensure_ascii=False, indent=1
            ),
        )

    def _allowed_counts(self, articles) -> set[int]:
        """Contagens que o texto pode citar: métricas, séries e números das fontes."""
        scope = self.ctx.scope
        charts = self.ctx.charts[scope]
        allowed = {p.cases for p in [*charts.daily, *charts.monthly]}
        for metric in self.ctx.metrics[scope].metrics:
            allowed |= {metric.numerator, metric.denominator}
            for caveat in metric.caveats:
                allowed |= numbers_in(caveat)
        for article in articles:
            allowed |= numbers_in(f"{article.title} {article.summary}")
        return allowed

    def _series_percentages(self) -> set[float]:
        """Variações mês a mês que o LLM pode citar ao descrever a tendência."""
        monthly = self.ctx.charts[self.ctx.scope].monthly
        allowed = set()
        for previous, current in zip(monthly, monthly[1:], strict=False):
            if previous.cases:
                allowed.add(round((current.cases - previous.cases) / previous.cases * 100, 1))
        return allowed


def generate_report(
    uf: str | None = None,
    focus: str | None = None,
    *,
    settings: Settings | None = None,
    llm_client=None,
) -> tuple[Path, AuditTrail]:
    """Ponto de entrada: valida o pedido, executa o grafo e devolve o relatório e a auditoria."""
    settings = settings or get_settings()
    audit = AuditTrail(settings.logs_dir)
    try:
        request = validate_request(uf, focus)
    except GuardrailViolation as exc:
        audit.record("guardrail", "input_validation", decision="rejected", reason=str(exc))
        raise
    audit.record(
        "run_started",
        "cli",
        request=request,
        provider=settings.llm_provider,
        model=settings.model_name,
        reporting_lag_days=settings.reporting_lag_days,
    )

    llm = create_provider(settings, audit, llm_client)
    with connect_readonly(settings.db_path) as con:
        period = get_report_period(con, settings.reporting_lag_days)
    ctx = RunContext(
        request=request,
        audit=audit,
        period=period,
        output_dir=settings.outputs_dir / audit.run_id,
        db_path=settings.db_path,
        model_label=f"{settings.llm_provider} / {settings.model_name}",
    )
    agent = SragReportAgent(ctx, llm, settings)
    try:
        final_state = agent.graph.invoke({}, {"recursion_limit": 4 * settings.max_agent_steps})
    except Exception as exc:
        audit.record("run_failed", "agent", error=f"{type(exc).__name__}: {exc}")
        raise
    audit.record("run_finished", "agent", report=final_state["report_path"])
    return final_state["report_path"], audit
