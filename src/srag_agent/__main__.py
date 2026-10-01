"""Linha de comando: gera um relatório de SRAG.

Exemplos:
    python -m srag_agent
    python -m srag_agent --uf SP
    python -m srag_agent --uf RJ --foco "Como está a pressão sobre UTIs pediátricas?"
"""

import argparse
import logging
import sys

from srag_agent.agent import generate_report
from srag_agent.guardrails import GuardrailViolation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera um relatório de SRAG com IA Generativa.")
    parser.add_argument("--uf", help="Sigla da UF (padrão: Brasil)")
    parser.add_argument("--foco", help="Pergunta ou foco opcional para a análise")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        report_path, audit = generate_report(uf=args.uf, focus=args.foco)
    except GuardrailViolation as exc:
        print(f"Pedido bloqueado pelos guardrails: {exc}", file=sys.stderr)
        return 2
    print(f"Relatório: {report_path}")
    print(f"Auditoria: {audit.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
