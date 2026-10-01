# SRAG Insight Agent

Prova de Conceito de uma solução de IA Generativa que gera, de forma automatizada, relatórios sobre
Síndrome Respiratória Aguda Grave (SRAG). Ela combina métricas calculadas sobre os dados oficiais do
[Open DATASUS](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026) com notícias em tempo real
para explicar o cenário atual.

> 🚧 Em construção. Este README será completado com arquitetura, guardrails, governança e decisões
> de projeto.

## Requisitos

- Python 3.11+
- Chave de API da Anthropic (Claude)

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # e preencha ANTHROPIC_API_KEY
```

## Dados

```bash
python -m srag_agent.data.download   # baixa os Parquet de 2025 e 2026 + dicionário de dados
python -m srag_agent.data.etl        # gera data/processed/srag.duckdb (tabela anonimizada)
```

O ETL lê apenas as colunas necessárias (allowlist em `src/srag_agent/data/etl.py`), converte idade em
faixa etária, restringe a localização à UF, descarta identificadores e campos de texto livre e registra
as contagens de limpeza na tabela `etl_quality`.

## Testes

```bash
pytest
```
