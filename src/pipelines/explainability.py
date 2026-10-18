"""Etapa 4 — explicabilidade: SHAP global e por grupo."""

import logging

import polars as pl

from config.settings import Settings
from constants.artifacts import GRADIENT_BOOSTING_MODEL_NAME, MODEL_FILES, TEST_PREDICTIONS_FILE
from constants.columns import SENSITIVE_COLUMNS
from constants.labels import QUESTION_LABELS
from data.loader import read_parquet
from data.sampler import sample_rows
from explainability.shap_analysis import (
    compute_shap_values,
    summarize_global_importance,
    summarize_shap_by_group,
)
from features.selection import select_feature_columns
from models.persistence import load_model
from preprocessing.encoding import encode_ordinal
from utils.serialization import write_table
from visualization.plots import (
    plot_shap_beeswarm,
    plot_shap_group_heatmap,
    plot_shap_importance,
    save_figure,
)

logger = logging.getLogger(__name__)


def run_explainability(settings: Settings) -> pl.DataFrame:
    """Calcula o SHAP do gradient boosting no conjunto de teste (global e por grupo).

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    pl.DataFrame
        Importância global por variável (``mean_abs_shap``).

    Raises
    ------
    FileNotFoundError
        Se o modelo ou as predições de teste não existirem (execute ``train``).

    Notes
    -----
    O SHAP é calculado em uma amostra aleatória do **teste** (nunca do treino),
    e a importância por grupo usa a mesma amostra, para comparar grupos sob o
    mesmo modelo. SHAP descreve o modelo, não relações causais.
    """
    explain = settings.explainability
    tables_dir = settings.paths.resolved("tables_dir")
    figures_dir = settings.paths.resolved("figures_dir")
    seed = settings.project.random_seed

    test = read_parquet(settings.paths.resolved("gold_dir") / TEST_PREDICTIONS_FILE)
    sample = sample_rows(test, explain.sample_size, seed)
    features = select_feature_columns()
    labels = [QUESTION_LABELS[name] for name in features]
    matrix = encode_ordinal(sample, features)
    model_path = settings.paths.resolved("models_dir") / MODEL_FILES[GRADIENT_BOOSTING_MODEL_NAME]
    shap_values = compute_shap_values(load_model(model_path), matrix)

    top = explain.top_features
    importance = summarize_global_importance(shap_values, features)
    write_table(importance, tables_dir / "shap_global.csv")
    save_figure(plot_shap_importance(importance, top), "shap_importance", figures_dir)
    save_figure(plot_shap_beeswarm(shap_values, matrix, labels, top), "shap_beeswarm", figures_dir)

    by_group_frames = []
    for attribute in SENSITIVE_COLUMNS:
        by_group = summarize_shap_by_group(
            shap_values,
            features,
            sample[attribute].to_numpy(),
            attribute,
            settings.fairness.min_group_size,
        )
        if by_group.is_empty():
            continue
        by_group_frames.append(by_group)
        save_figure(
            plot_shap_group_heatmap(by_group, attribute, explain.top_features),
            f"shap_by_{attribute}",
            figures_dir,
        )
    if by_group_frames:
        write_table(pl.concat(by_group_frames), tables_dir / "shap_by_group.csv")
    logger.info("Explicabilidade concluída (%d linhas explicadas)", sample.height)
    return importance
