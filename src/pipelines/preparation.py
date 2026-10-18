"""Etapa 2 — preparação: silver → gold (presença, alvo, grupos e ausentes)."""

import logging
from collections.abc import Sequence
from pathlib import Path

from config.settings import Settings
from constants.artifacts import GOLD_FILE, GOLD_MANIFEST_FILE
from constants.columns import YEAR_COLUMN
from data.loader import scan_silver
from data.writer import write_parquet
from exceptions.data import InsufficientDataError
from features.engineering import build_gold_frame
from preprocessing.filtering import (
    apply_filter_steps,
    build_filter_steps,
    compute_participation_funnel,
)
from schemas.dataset import validate_gold
from utils.hashing import hash_file
from utils.serialization import write_json, write_table

logger = logging.getLogger(__name__)


def run_preparation(settings: Settings, years: Sequence[int] | None = None) -> Path:
    """Gera a tabela analítica (gold) a partir da camada silver.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    years : Sequence[int] or None, optional
        Anos a incluir; ``None`` usa todas as partições silver disponíveis.

    Returns
    -------
    Path
        Caminho do Parquet gold.

    Raises
    ------
    FileNotFoundError
        Se não houver partições silver.
    InsufficientDataError
        Se nenhum participante válido restar após os filtros.
    DataValidationError
        Se a tabela gold violar o contrato de dados.

    Notes
    -----
    Mantém apenas participantes presentes nas quatro provas objetivas, com
    redação válida e todas as notas informadas. O funil de seleção é gravado
    em ``reports/tables/participation_funnel.csv`` para documentar o viés de
    seleção (quem falta à prova não entra na análise).
    """
    silver = scan_silver(settings.paths.resolved("silver_dir"), years)
    steps = build_filter_steps(settings.preparation.exclude_trainees)
    funnel = compute_participation_funnel(silver, steps)
    write_table(funnel, settings.paths.resolved("tables_dir") / "participation_funnel.csv")

    gold = build_gold_frame(apply_filter_steps(silver, steps)).collect()
    if gold.is_empty():
        raise InsufficientDataError("Nenhum participante válido após os filtros de presença")
    validate_gold(gold)

    gold_path = write_parquet(gold, settings.paths.resolved("gold_dir") / GOLD_FILE)
    write_json(
        {
            "rows": gold.height,
            "years": sorted(gold[YEAR_COLUMN].unique().to_list()),
            "columns": gold.columns,
            "sha256": hash_file(gold_path),
        },
        settings.paths.resolved("gold_dir") / GOLD_MANIFEST_FILE,
    )
    logger.info("Gold gravada: %d linhas em %s", gold.height, gold_path)
    return gold_path
