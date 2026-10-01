"""Formato estruturado da análise escrita pelo LLM."""

from typing import Literal

from pydantic import BaseModel, Field

from srag_agent.guardrails import mask_pii

MetricKey = Literal["taxa_aumento_casos", "taxa_mortalidade", "taxa_ocupacao_uti", "taxa_vacinacao"]


class MetricCommentary(BaseModel):
    chave: MetricKey
    comentario: str = Field(description="O que o valor indica e como interpretá-lo")


class ReportAnalysis(BaseModel):
    resumo_executivo: str = Field(description="3 a 5 frases com a leitura geral do cenário")
    analise_metricas: list[MetricCommentary]
    tendencia_casos: str = Field(description="Leitura das séries diária e mensal")
    contexto_noticias: str = Field(description="Como as notícias ajudam a explicar o cenário")
    noticias_citadas: list[int] = Field(description="ids das notícias usadas")
    pontos_de_atencao: list[str] = Field(description="Pontos para vigilância e gestão")

    def full_text(self) -> str:
        return "\n".join(
            [
                self.resumo_executivo,
                self.tendencia_casos,
                self.contexto_noticias,
                *(c.comentario for c in self.analise_metricas),
                *self.pontos_de_atencao,
            ]
        )

    def masked(self) -> "ReportAnalysis":
        """Cópia com dados pessoais mascarados em todos os campos de texto."""

        def mask(text: str) -> str:
            return mask_pii(text)[0]

        return self.model_copy(
            update={
                "resumo_executivo": mask(self.resumo_executivo),
                "tendencia_casos": mask(self.tendencia_casos),
                "contexto_noticias": mask(self.contexto_noticias),
                "analise_metricas": [
                    c.model_copy(update={"comentario": mask(c.comentario)})
                    for c in self.analise_metricas
                ],
                "pontos_de_atencao": [mask(p) for p in self.pontos_de_atencao],
            }
        )
