"""Etapa 5 — equidade: desempenho e erro por renda, raça/cor, escola e região."""

import logging

import polars as pl

from config.settings import Settings
from constants.artifacts import MODEL_NAMES, PREDICTION_PREFIX, TEST_PREDICTIONS_FILE
from constants.columns import SENSITIVE_COLUMNS, TARGET_COLUMN
from data.loader import read_parquet
from exceptions.configuration import ConfigurationError
from fairness.group_metrics import analyze_group_equity
from utils.serialization import write_table
from visualization.plots import plot_group_estimates, save_figure

logger = logging.getLogger(__name__)


def _validate_reference_groups(settings: Settings) -> None:
    """Garante que todo atributo sensível tenha grupo de referência configurado."""
    reference_groups = settings.fairness.reference_groups
    missing = [name for name in SENSITIVE_COLUMNS if name not in reference_groups]
    if missing:
        raise ConfigurationError(f"Grupo de referência ausente para: {missing}")


def _analyze_model(
    settings: Settings, test: pl.DataFrame, model_name: str
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Analisa um modelo em todos os atributos sensíveis e salva os gráficos."""
    fairness = settings.fairness
    y_true = test[TARGET_COLUMN].to_numpy()
    y_pred = test[f"{PREDICTION_PREFIX}{model_name}"].to_numpy()
    results: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    for attribute in SENSITIVE_COLUMNS:
        metrics, gaps = analyze_group_equity(
            y_true,
            y_pred,
            test[attribute].to_numpy(),
            attribute,
            fairness.reference_groups[attribute],
            fairness.n_bootstrap,
            fairness.confidence_level,
            fairness.min_group_size,
            settings.project.random_seed,
        )
        if not metrics.is_empty():
            save_figure(
                plot_group_estimates(metrics, attribute),
                f"fairness_{model_name}_{attribute}",
                settings.paths.resolved("figures_dir"),
            )
        results.append(
            (
                metrics.with_columns(pl.lit(model_name).alias("model")),
                gaps.with_columns(pl.lit(model_name).alias("model")),
            )
        )
    return results


def run_fairness(settings: Settings) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Mede diferenças de desempenho e de erro entre grupos, com IC bootstrap.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(metrics, gaps)`` de todos os modelos e atributos, gravados em
        ``reports/tables/fairness_metrics.csv`` e ``fairness_gaps.csv``.

    Raises
    ------
    FileNotFoundError
        Se as predições de teste não existirem (execute ``train``).
    ConfigurationError
        Se faltar o grupo de referência de algum atributo na configuração.

    Notes
    -----
    A análise usa o conjunto de teste (linhas nunca vistas no treino). As
    diferenças são ``grupo - referência``; ``mean_error`` positivo significa
    que o modelo superestima a nota do grupo. Diferenças observadas descrevem
    o modelo e os dados, não causas.
    """
    _validate_reference_groups(settings)
    test = read_parquet(settings.paths.resolved("gold_dir") / TEST_PREDICTIONS_FILE)
    metrics_frames: list[pl.DataFrame] = []
    gap_frames: list[pl.DataFrame] = []
    for model_name in MODEL_NAMES:
        for metrics, gaps in _analyze_model(settings, test, model_name):
            metrics_frames.append(metrics)
            gap_frames.append(gaps)

    all_metrics = pl.concat(metrics_frames, how="diagonal")
    all_gaps = pl.concat(gap_frames, how="diagonal")
    tables_dir = settings.paths.resolved("tables_dir")
    write_table(all_metrics, tables_dir / "fairness_metrics.csv")
    write_table(all_gaps, tables_dir / "fairness_gaps.csv")
    logger.info("Equidade concluída: %d grupos avaliados", all_metrics.height)
    return all_metrics, all_gaps
