"""ETL: arquivos brutos do SIVEP-Gripe (Open DATASUS) -> tabela analítica anonimizada no DuckDB.

Princípios:
- Seleção por *allowlist*: só entram no banco as colunas necessárias às métricas.
  Identificadores diretos (nº da notificação, data de nascimento), localização fina
  (município, unidade de saúde), campos de texto livre e lotes de vacina nunca saem
  do arquivo bruto.
- A idade exata vira faixa etária e a localização fica restrita à UF, reduzindo o
  risco de reidentificação.
- Códigos do dicionário de dados viram rótulos legíveis; "9-Ignorado" e vazios viram NULL.
- Toda regra de limpeza é contabilizada na tabela `etl_quality`, para auditoria.

Uso:
    python -m srag_agent.data.etl
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from srag_agent.config import get_settings

logger = logging.getLogger(__name__)

CASES_TABLE = "srag_cases"
QUALITY_TABLE = "etl_quality"
METADATA_TABLE = "etl_metadata"

# Colunas lidas do arquivo bruto e o motivo de cada uma estar aqui.
SELECTED_COLUMNS: dict[str, str] = {
    "NU_NOTIFIC": "usado só para deduplicar; descartado antes da gravação",
    "CO_MUN_NOT": "usado só para deduplicar; descartado antes da gravação",
    "DT_NOTIFIC": "data da notificação",
    "DT_SIN_PRI": "data de início dos sintomas (data de referência dos casos)",
    "DT_DIGITA": "data de digitação (mede o atraso de notificação)",
    "DT_INTERNA": "data da internação",
    "DT_ENTUTI": "data de entrada na UTI",
    "DT_SAIDUTI": "data de saída da UTI",
    "DT_EVOLUCA": "data da alta ou do óbito",
    "SG_UF_NOT": "UF de notificação",
    "SG_UF": "UF de residência",
    "CS_SEXO": "sexo",
    "NU_IDADE_N": "idade (convertida em faixa etária)",
    "TP_IDADE": "unidade da idade: 1-dias, 2-meses, 3-anos",
    "HOSPITAL": "houve internação",
    "UTI": "internado em UTI",
    "SUPORT_VEN": "uso de suporte ventilatório",
    "CLASSI_FIN": "classificação final do caso (agente etiológico)",
    "EVOLUCAO": "evolução do caso: cura, óbito por SRAG, óbito por outras causas",
    "VACINA_COV": "recebeu vacina contra COVID-19",
    "VACINA": "recebeu vacina contra gripe na última campanha",
}

# Faixas etárias (anos) usadas no lugar da idade exata.
AGE_BANDS_SQL = """
    CASE
        WHEN idade_anos IS NULL OR idade_anos < 0 OR idade_anos > 120 THEN NULL
        WHEN idade_anos < 2 THEN '0-1'
        WHEN idade_anos < 5 THEN '2-4'
        WHEN idade_anos < 12 THEN '5-11'
        WHEN idade_anos < 18 THEN '12-17'
        WHEN idade_anos < 30 THEN '18-29'
        WHEN idade_anos < 40 THEN '30-39'
        WHEN idade_anos < 50 THEN '40-49'
        WHEN idade_anos < 60 THEN '50-59'
        WHEN idade_anos < 70 THEN '60-69'
        WHEN idade_anos < 80 THEN '70-79'
        ELSE '80+'
    END
"""


def _yes_no(column: str) -> str:
    """Campos 1-Sim / 2-Não / 9-Ignorado -> BOOLEAN (Ignorado e vazio viram NULL)."""
    return f"CASE TRIM(CAST({column} AS VARCHAR)) WHEN '1' THEN TRUE WHEN '2' THEN FALSE END"


def _code(column: str) -> str:
    """Normaliza códigos que chegam como texto ou decimal ('1', '1.0', 1) para texto inteiro."""
    return f"TRY_CAST(TRY_CAST({column} AS DOUBLE) AS INTEGER)"


def _valid_date(column: str) -> str:
    """Datas fora do intervalo [início dos sintomas, data de digitação] viram NULL."""
    return (
        f"CASE WHEN {column} IS NULL THEN NULL "
        f"WHEN {column} < DT_SIN_PRI OR {column} > DT_DIGITA THEN NULL "
        f"ELSE CAST({column} AS DATE) END"
    )


def _source_glob(raw_dir: Path, source_files: list[str]) -> list[str]:
    paths = [raw_dir / Path(name).name for name in source_files]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos brutos ausentes em {raw_dir}: {missing}. "
            "Rode antes: python -m srag_agent.data.download"
        )
    return [p.as_posix() for p in paths]


def build_database(db_path: Path | None = None) -> dict[str, int]:
    """Executa o ETL completo e retorna as contagens de qualidade (também gravadas no banco)."""
    settings = get_settings()
    db_path = db_path or settings.db_path
    sources = _source_glob(settings.raw_dir, settings.source_files)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ", ".join(SELECTED_COLUMNS)
    # Views não aceitam parâmetros preparados; os caminhos vêm da configuração, não do usuário.
    source_list = ", ".join("'" + s.replace("'", "''") + "'" for s in sources)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW raw AS
            SELECT {columns}, filename AS source_file
            FROM read_parquet([{source_list}], union_by_name = true, filename = true)
            """
        )
        raw_rows = con.execute("SELECT count(*) FROM raw").fetchone()[0]

        # 1) Deduplicação: o mesmo caso pode aparecer em mais de um arquivo anual.
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW deduplicated AS
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY NU_NOTIFIC, CO_MUN_NOT, DT_NOTIFIC
                    ORDER BY DT_DIGITA DESC NULLS LAST
                ) AS rn
                FROM raw
            ) WHERE rn = 1
            """
        )
        dedup_rows = con.execute("SELECT count(*) FROM deduplicated").fetchone()[0]

        # 2) Registros sem data de sintomas ou com datas incoerentes não entram nas séries.
        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW valid AS
            SELECT * FROM deduplicated
            WHERE DT_SIN_PRI IS NOT NULL
              AND DT_NOTIFIC IS NOT NULL
              AND DT_SIN_PRI <= DT_NOTIFIC
              AND (DT_DIGITA IS NULL OR DT_SIN_PRI <= DT_DIGITA)
            """
        )
        valid_rows = con.execute("SELECT count(*) FROM valid").fetchone()[0]

        # 3) Tabela final: só colunas anonimizadas e com rótulos legíveis.
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {CASES_TABLE} AS
            WITH typed AS (
                SELECT *,
                    CASE {_code("TP_IDADE")}
                        WHEN 1 THEN NU_IDADE_N / 365.25
                        WHEN 2 THEN NU_IDADE_N / 12.0
                        WHEN 3 THEN NU_IDADE_N
                    END AS idade_anos
                FROM valid
            )
            SELECT
                row_number() OVER (ORDER BY DT_SIN_PRI, DT_NOTIFIC) AS case_id,
                CAST(DT_SIN_PRI AS DATE) AS dt_sintomas,
                CAST(DT_NOTIFIC AS DATE) AS dt_notificacao,
                CAST(DT_DIGITA AS DATE) AS dt_digitacao,
                {_valid_date("DT_INTERNA")} AS dt_internacao,
                {_valid_date("DT_ENTUTI")} AS dt_entrada_uti,
                {_valid_date("DT_SAIDUTI")} AS dt_saida_uti,
                {_valid_date("DT_EVOLUCA")} AS dt_evolucao,
                NULLIF(TRIM(SG_UF_NOT), '') AS uf_notificacao,
                NULLIF(TRIM(SG_UF), '') AS uf_residencia,
                CASE TRIM(CS_SEXO) WHEN 'M' THEN 'Masculino' WHEN 'F' THEN 'Feminino' END AS sexo,
                {AGE_BANDS_SQL} AS faixa_etaria,
                {_yes_no("HOSPITAL")} AS hospitalizado,
                {_yes_no("UTI")} AS internado_uti,
                CASE {_code("SUPORT_VEN")}
                    WHEN 1 THEN 'Invasivo' WHEN 2 THEN 'Não invasivo' WHEN 3 THEN 'Não'
                END AS suporte_ventilatorio,
                CASE {_code("CLASSI_FIN")}
                    WHEN 1 THEN 'Influenza'
                    WHEN 2 THEN 'Outro vírus respiratório'
                    WHEN 3 THEN 'Outro agente etiológico'
                    WHEN 4 THEN 'Não especificado'
                    WHEN 5 THEN 'COVID-19'
                END AS classificacao_final,
                CASE {_code("EVOLUCAO")}
                    WHEN 1 THEN 'Cura'
                    WHEN 2 THEN 'Óbito por SRAG'
                    WHEN 3 THEN 'Óbito por outras causas'
                END AS evolucao,
                {_yes_no("VACINA_COV")} AS vacinado_covid,
                {_yes_no("VACINA")} AS vacinado_gripe,
                regexp_extract(source_file, '[^/\\\\]+$') AS arquivo_origem
            FROM typed
            """
        )

        quality = {
            "linhas_brutas": raw_rows,
            "duplicatas_removidas": raw_rows - dedup_rows,
            "datas_invalidas_removidas": dedup_rows - valid_rows,
            "linhas_finais": valid_rows,
            **_null_counts(con),
        }
        _write_quality(con, quality)
        _write_metadata(con, sources)

    logger.info("ETL concluído: %s", quality)
    return quality


def _null_counts(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Quantos registros ficaram sem informação (vazio ou 'Ignorado') nos campos das métricas."""
    fields = ["evolucao", "internado_uti", "vacinado_covid", "vacinado_gripe", "faixa_etaria"]
    exprs = ", ".join(f"count(*) FILTER (WHERE {f} IS NULL)" for f in fields)
    values = con.execute(f"SELECT {exprs} FROM {CASES_TABLE}").fetchone()
    return {f"sem_informacao_{f}": v for f, v in zip(fields, values, strict=True)}


def _write_quality(con: duckdb.DuckDBPyConnection, quality: dict[str, int]) -> None:
    con.execute(f"CREATE OR REPLACE TABLE {QUALITY_TABLE} (indicador VARCHAR, valor BIGINT)")
    con.executemany(f"INSERT INTO {QUALITY_TABLE} VALUES (?, ?)", list(quality.items()))


def _write_metadata(con: duckdb.DuckDBPyConnection, sources: list[str]) -> None:
    con.execute(f"CREATE OR REPLACE TABLE {METADATA_TABLE} (chave VARCHAR, valor VARCHAR)")
    max_digitacao = con.execute(f"SELECT max(dt_digitacao) FROM {CASES_TABLE}").fetchone()[0]
    con.executemany(
        f"INSERT INTO {METADATA_TABLE} VALUES (?, ?)",
        [
            ("etl_executado_em", datetime.now(UTC).isoformat(timespec="seconds")),
            ("arquivos_origem", ", ".join(Path(s).name for s in sources)),
            ("data_mais_recente_digitacao", str(max_digitacao)),
        ],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_database()


if __name__ == "__main__":
    main()
