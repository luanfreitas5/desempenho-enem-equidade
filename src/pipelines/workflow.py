"""Encadeia as etapas do pipeline, da ingestão à análise de equidade."""

import logging
from collections.abc import Callable, Sequence

from config.settings import Settings
from pipelines.explainability import run_explainability
from pipelines.fairness import run_fairness
from pipelines.ingestion import run_ingestion
from pipelines.preparation import run_preparation
from pipelines.training import run_training

logger = logging.getLogger(__name__)

STAGE_NAMES: tuple[str, ...] = ("ingest", "prepare", "train", "explain", "fairness")


def build_stage_runners(
    settings: Settings, years: Sequence[int] | None, force: bool
) -> dict[str, Callable[[], object]]:
    """Mapeia o nome de cada etapa à função que a executa.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    years : Sequence[int] or None
        Anos a processar nas etapas de dados; ``None`` usa a configuração.
    force : bool
        Se ``True``, refaz a ingestão ignorando o cache.

    Returns
    -------
    dict[str, Callable[[], object]]
        Funções sem argumentos, na ordem do pipeline.
    """
    return {
        "ingest": lambda: run_ingestion(settings, years, force),
        "prepare": lambda: run_preparation(settings, years),
        "train": lambda: run_training(settings),
        "explain": lambda: run_explainability(settings),
        "fairness": lambda: run_fairness(settings),
    }


def run_all(settings: Settings, years: Sequence[int] | None = None, force: bool = False) -> None:
    """Executa todas as etapas em ordem.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    years : Sequence[int] or None, optional
        Anos a processar; ``None`` usa ``settings.ingestion.years``.
    force : bool, optional
        Se ``True``, refaz a ingestão ignorando o cache, by default False.

    Examples
    --------
    >>> run_all(load_settings())  # doctest: +SKIP
    """
    runners = build_stage_runners(settings, years, force)
    for stage in STAGE_NAMES:
        logger.info("=== Etapa: %s ===", stage)
        runners[stage]()
