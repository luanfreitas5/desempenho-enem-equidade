"""Ponto de entrada do pipeline (CLI).

Uso
---
    uv run python src/main.py all
    uv run python src/main.py ingest --years 2023 --force
    uv run python src/main.py train
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

# `src/` é a raiz de importação: garante o path ao executar `python src/main.py`.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.environment import seed_everything
from config.logging import configure_logging
from config.settings import load_settings
from exceptions.base import EnemEquityError
from pipelines.workflow import STAGE_NAMES, build_stage_runners, run_all
from visualization.theme import apply_theme

logger = logging.getLogger("main")


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da linha de comando.

    Returns
    -------
    argparse.ArgumentParser
        Parser com a etapa (``stage``) e as opções ``--config-dir``,
        ``--years``, ``--force`` e ``--log-level``.
    """
    parser = argparse.ArgumentParser(
        prog="desempenho-enem-equidade",
        description="Modelo explicável da nota do ENEM e análise de equidade.",
    )
    parser.add_argument(
        "stage",
        choices=[*STAGE_NAMES, "all"],
        help="Etapa a executar ('all' roda o pipeline completo).",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None, help="Pasta dos YAML de configuração."
    )
    parser.add_argument("--years", type=int, nargs="+", default=None, help="Anos a processar.")
    parser.add_argument("--force", action="store_true", help="Ignora o cache e refaz a ingestão.")
    parser.add_argument("--log-level", default=None, help="Nível de log (sobrescreve o YAML).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a etapa pedida na linha de comando.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Argumentos (``None`` usa ``sys.argv``).

    Returns
    -------
    int
        Código de saída: 0 em sucesso e 1 em erro do projeto.
    """
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config_dir)
    configure_logging(
        args.log_level or settings.logging.level,
        settings.paths.resolved("logs_dir"),
        settings.logging.backup_count,
    )
    seed_everything(settings.project.random_seed)
    apply_theme()
    try:
        if args.stage == "all":
            run_all(settings, args.years, args.force)
        else:
            build_stage_runners(settings, args.years, args.force)[args.stage]()
    except EnemEquityError:
        logger.exception("Falha na etapa '%s'", args.stage)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
