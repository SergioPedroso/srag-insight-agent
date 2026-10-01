"""Guardrails determinísticos aplicados antes e depois do LLM.

Camadas:
1. Entrada: o pedido de relatório é estruturado (UF validada contra lista fechada; pergunta
   opcional limitada em tamanho, restrita ao tema e checada contra injeção de instruções).
2. Dados: o agente não escreve SQL; as tools só executam consultas fixas em banco read-only.
3. Conteúdo externo: notícias com padrões de injeção são descartadas e dados pessoais são
   mascarados antes de chegarem ao LLM.
4. Saída: todo percentual citado pelo LLM precisa bater com um valor calculado ou presente
   nas fontes; o texto final passa por mascaramento de dados pessoais.
"""

import re
import unicodedata

from pydantic import BaseModel, Field, ValidationError, field_validator

from srag_agent.metrics import MetricsSnapshot, normalize_uf
from srag_agent.news import NewsArticle

MAX_FOCUS_CHARS = 300
PERCENT_TOLERANCE_PP = 0.15

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"ignore (as )?instru[cç][oõ]es",
    r"desconsidere (as |todas as )?(instru[cç][oõ]es|regras)",
    r"system prompt|prompt do sistema",
    r"you are now|voc[eê] agora [eé]",
    r"act as|aja como|finja (ser|que)",
    r"</?(system|assistant|user|instructions?)>",
    r"reveal (your|the) (instructions|prompt)|revele (suas|as) instru",
    r"jailbreak|developer mode|modo desenvolvedor",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

TOPIC_KEYWORDS = [
    "srag", "sindrome respiratoria", "respirator", "gripe", "influenza", "covid", "virus",
    "vsr", "rinovirus", "metapneumovirus", "caso", "obito", "morte", "mortalidade",
    "letalidade", "uti", "internac", "hospital", "vacina", "surto", "epidemi", "crianca",
    "idoso", "tendencia", "aumento", "queda", "estado", "regiao",
]  # fmt: skip

PII_PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNS": re.compile(r"\b[1-9]\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "TELEFONE": re.compile(r"(?<!\d)(?:\(?\d{2}\)?\s?)?9?\d{4}-?\d{4}(?!\d)"),
}

PERCENT_RE = re.compile(r"(-?\d{1,3}(?:[.,]\d+)?)\s?%")


class GuardrailViolation(ValueError):
    """Pedido bloqueado por um guardrail de entrada."""


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def contains_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(_normalize(text)) or _INJECTION_RE.search(text))


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Substitui CPF, CNS, e-mail e telefone por marcadores. Retorna o texto e as contagens."""
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label} REMOVIDO]", text)
        if n:
            counts[label] = n
    return text, counts


class ReportRequest(BaseModel):
    """Pedido de relatório. Todo campo é validado antes de qualquer chamada ao LLM."""

    uf: str | None = Field(default=None, description="Sigla da UF; vazio = Brasil")
    focus: str | None = Field(default=None, description="Pergunta ou foco opcional para a análise")

    @field_validator("uf")
    @classmethod
    def _validate_uf(cls, value: str | None) -> str | None:
        try:
            return normalize_uf(value)
        except ValueError as exc:
            raise GuardrailViolation(str(exc)) from exc

    @field_validator("focus")
    @classmethod
    def _validate_focus(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if len(value) > MAX_FOCUS_CHARS:
            raise GuardrailViolation(f"O foco deve ter no máximo {MAX_FOCUS_CHARS} caracteres.")
        if contains_injection(value):
            raise GuardrailViolation("O foco contém instruções não permitidas.")
        normalized = _normalize(value)
        if not any(keyword in normalized for keyword in TOPIC_KEYWORDS):
            raise GuardrailViolation(
                "O foco precisa estar relacionado a SRAG e às métricas do relatório."
            )
        masked, counts = mask_pii(value)
        if counts:
            raise GuardrailViolation("O foco não pode conter dados pessoais.")
        return masked


def validate_request(uf: str | None, focus: str | None) -> ReportRequest:
    """Cria o pedido validado; qualquer violação vira `GuardrailViolation` com motivo legível."""
    try:
        return ReportRequest(uf=uf, focus=focus)
    except ValidationError as exc:
        reasons = "; ".join(str(e.get("ctx", {}).get("error", e["msg"])) for e in exc.errors())
        raise GuardrailViolation(reasons) from exc


class NewsScreening(BaseModel):
    accepted: list[NewsArticle]
    rejected: list[dict]


def screen_news(articles: list[NewsArticle]) -> NewsScreening:
    """Descarta notícias com tentativa de injeção e mascara dados pessoais nas demais."""
    accepted, rejected = [], []
    for article in articles:
        text = f"{article.title} {article.summary}"
        if contains_injection(text):
            rejected.append({"title": article.title, "reason": "possível injeção de instruções"})
            continue
        title, _ = mask_pii(article.title)
        summary, _ = mask_pii(article.summary)
        accepted.append(article.model_copy(update={"title": title, "summary": summary}))
    return NewsScreening(accepted=accepted, rejected=rejected)


def _parse_percent(raw: str) -> float:
    return float(raw.replace(",", "."))


def allowed_percentages(snapshot: MetricsSnapshot, news: list[NewsArticle]) -> set[float]:
    """Percentuais que o texto pode citar: métricas, limitações e números das notícias."""
    allowed: set[float] = set()
    for metric in snapshot.metrics:
        if metric.value is not None:
            allowed.add(round(metric.value * 100, 1))
            allowed.add(round(abs(metric.value) * 100, 1))
        for caveat in metric.caveats:
            allowed.update(_parse_percent(m) for m in PERCENT_RE.findall(caveat))
    for article in news:
        allowed.update(
            _parse_percent(m) for m in PERCENT_RE.findall(f"{article.title} {article.summary}")
        )
    return allowed


def find_unverified_percentages(text: str, allowed: set[float]) -> list[str]:
    """Percentuais do texto sem correspondência (± tolerância) nos valores permitidos."""
    unverified = []
    for raw in PERCENT_RE.findall(text):
        value = abs(_parse_percent(raw))
        if not any(abs(value - abs(a)) <= PERCENT_TOLERANCE_PP for a in allowed):
            unverified.append(f"{raw}%")
    return unverified
