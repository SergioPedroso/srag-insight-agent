"""Busca de notícias recentes sobre SRAG via Google News RSS (sem chave de API).

Notícias são conteúdo externo e não confiável: aqui elas são reduzidas a campos simples
(título, fonte, data, link, resumo em texto puro e truncado). A checagem contra
injeção de instruções acontece em `guardrails`, antes de o texto chegar ao LLM.
"""

import html
import logging
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import httpx
from pydantic import BaseModel

from srag_agent.geo import UF_NAMES

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
DEFAULT_QUERY = '"SRAG" OR "síndrome respiratória aguda grave"'
MAX_SUMMARY_CHARS = 400
TAG_RE = re.compile(r"<[^>]+>")
SPACES_RE = re.compile(r"\s+")


class NewsArticle(BaseModel):
    title: str
    source: str
    published_at: datetime | None
    url: str
    summary: str


def _clean_text(raw: str, limit: int) -> str:
    text = SPACES_RE.sub(" ", html.unescape(TAG_RE.sub(" ", raw or ""))).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _published(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    return datetime(*parsed[:6], tzinfo=UTC) if parsed else None


def parse_feed(content: bytes | str, *, max_age_days: int, limit: int) -> list[NewsArticle]:
    feed = feedparser.parse(content)
    min_date = datetime.now(UTC) - timedelta(days=max_age_days)
    seen_titles: set[str] = set()
    articles: list[NewsArticle] = []
    for entry in feed.entries:
        published = _published(entry)
        if published and published < min_date:
            continue
        source = entry.get("source", {}).get("title", "") or ""
        title = _clean_text(entry.get("title", ""), 200)
        # O Google News acrescenta " - Fonte" ao título; remove para deduplicar.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        key = title.casefold()
        if not title or key in seen_titles:
            continue
        seen_titles.add(key)
        summary = _clean_text(entry.get("summary", ""), MAX_SUMMARY_CHARS)
        # No Google News o resumo costuma repetir título + fonte; nesse caso não agrega nada.
        if summary.casefold().startswith(key):
            summary = ""
        articles.append(
            NewsArticle(
                title=title,
                source=_clean_text(source, 80),
                published_at=published,
                url=entry.get("link", ""),
                summary=summary,
            )
        )
    articles.sort(key=lambda a: a.published_at or datetime.min.replace(tzinfo=UTC), reverse=True)
    return articles[:limit]


def build_query(uf: str | None = None) -> str:
    """Consulta fixa sobre SRAG; o escopo só pode ser uma UF já validada, nunca texto livre."""
    return f'({DEFAULT_QUERY}) "{UF_NAMES[uf]}"' if uf else DEFAULT_QUERY


def search_news(
    query: str = DEFAULT_QUERY, *, max_age_days: int = 30, limit: int = 10, timeout: float = 20
) -> list[NewsArticle]:
    url = GOOGLE_NEWS_RSS.format(query=quote_plus(f"{query} when:{max_age_days}d"))
    response = httpx.get(
        url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "srag-insight-agent"}
    )
    response.raise_for_status()
    articles = parse_feed(response.content, max_age_days=max_age_days, limit=limit)
    logger.info("Notícias encontradas: %d", len(articles))
    return articles
