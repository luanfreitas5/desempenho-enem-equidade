"""Etapa 3 — modelagem: regressão linear (referência) e gradient boosting."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.settings import Settings
from constants.artifacts import (
    GOLD_FILE,
    GRADIENT_BOOSTING_MODEL_NAME,
    LINEAR_MODEL_NAME,
    MODEL_FILES,
    MODEL_NAMES,
    PREDICTION_PREFIX,
    TEST_PREDICTIONS_FILE,
)
from constants.columns import TARGET_COLUMN
from data.loader import read_parquet
from data.sampler import sample_rows
from data.splitter import split_train_test
from data.writer import write_parquet
from evaluation.comparison import corrected_resampled_ttest
from evaluation.metrics import (
    bootstrap_metric_difference,
    bootstrap_statistics,
    calculate_mae,
    calculate_r2,
    calculate_rmse,
    compute_percentile_interval,
    summarize_scores,
)
from features.selection import select_feature_columns
from models.base import Regressor
from models.gradient_boosting import build_gradient_boosting
from models.linear_baseline import LinearBaseline
from models.persistence import save_model
from preprocessing.encoding import encode_categorical, encode_ordinal
from training.cross_validation import FoldResult, cross_validate_model, results_to_frame
from utils.hashing import hash_file
from utils.serialization import write_json, write_table
from visualization.plots import plot_model_comparison, save_figure

logger = logging.getLogger(__name__)

CV_METRICS = ("rmse", "mae", "r2")
TEST_STATISTICS = {"rmse": calculate_rmse, "mae": calculate_mae, "r2": calculate_r2}


@dataclass(frozen=True)
class ModelingData:
    """Matrizes de treino/teste de cada modelo e alvos."""

    features: list[str]
    train: pl.DataFrame
    test: pl.DataFrame
    X_train: dict[str, np.ndarray]
    X_test: dict[str, np.ndarray]
    y_train: np.ndarray
    y_test: np.ndarray


def prepare_modeling_data(settings: Settings) -> tuple[ModelingData, Path]:
    """Carrega a gold, amostra, divide em treino/teste e codifica as features.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    tuple[ModelingData, Path]
        Dados de modelagem e o caminho do arquivo gold usado.

    Raises
    ------
    FileNotFoundError
        Se a tabela gold não existir.
    LeakageError
        Se alguma feature vazar o alvo.
    """
    seed = settings.project.random_seed
    gold_path = settings.paths.resolved("gold_dir") / GOLD_FILE
    features = select_feature_columns()
    sample = sample_rows(read_parquet(gold_path), settings.modeling.sample_size, seed)
    train, test = split_train_test(sample, settings.modeling.test_size, seed)
    data = ModelingData(
        features=features,
        train=train,
        test=test,
        X_train={
            LINEAR_MODEL_NAME: encode_categorical(train, features),
            GRADIENT_BOOSTING_MODEL_NAME: encode_ordinal(train, features),
        },
        X_test={
            LINEAR_MODEL_NAME: encode_categorical(test, features),
            GRADIENT_BOOSTING_MODEL_NAME: encode_ordinal(test, features),
        },
        y_train=train[TARGET_COLUMN].to_numpy(),
        y_test=test[TARGET_COLUMN].to_numpy(),
    )
    logger.info("Treino: %d linhas; teste: %d linhas", train.height, test.height)
    return data, gold_path


def build_model_factories(settings: Settings, features: list[str]) -> dict[str, Callable[[], Any]]:
    """Cria as fábricas de modelos não treinados.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    features : list[str]
        Nomes das features (usados pelo modelo linear nos coeficientes).

    Returns
    -------
    dict[str, Callable[[], Any]]
        Fábrica por nome de modelo.
    """
    seed = settings.project.random_seed
    params = settings.modeling.gradient_boosting
    return {
        LINEAR_MODEL_NAME: lambda: LinearBaseline(features),
        GRADIENT_BOOSTING_MODEL_NAME: lambda: build_gradient_boosting(params, seed),
    }


def summarize_cross_validation(
    results: dict[str, list[FoldResult]], confidence_level: float
) -> pl.DataFrame:
    """Resume a validação cruzada: média, desvio e IC por modelo e métrica.

    Parameters
    ----------
    results : dict[str, list[FoldResult]]
        Resultados por dobra de cada modelo.
    confidence_level : float
        Nível de confiança do IC (t de Student).

    Returns
    -------
    pl.DataFrame
        Colunas ``model``, ``metric``, ``mean``, ``std``, ``ci_low`` e ``ci_high``.
    """
    rows = []
    for model_name, folds in results.items():
        for metric in CV_METRICS:
            summary = summarize_scores([getattr(fold, metric) for fold in folds], confidence_level)
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": summary.mean,
                    "std": summary.std,
                    "ci_low": summary.ci_low,
                    "ci_high": summary.ci_high,
                }
            )
    return pl.DataFrame(rows)


def evaluate_on_test(
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> pl.DataFrame:
    """Avalia cada modelo no conjunto de teste, com IC bootstrap das métricas.

    Parameters
    ----------
    predictions : dict[str, np.ndarray]
        Previsões do teste por modelo.
    y_test : np.ndarray
        Notas observadas no teste.
    n_bootstrap : int
        Número de réplicas bootstrap.
    confidence_level : float
        Nível de confiança dos intervalos.
    seed : int
        Semente do bootstrap.

    Returns
    -------
    pl.DataFrame
        Uma linha por modelo, com ``rmse``, ``mae`` e ``r2`` e seus limites
        ``_low``/``_high``.
    """
    rows = []
    for model_name, predicted in predictions.items():
        replicates = bootstrap_statistics(y_test, predicted, TEST_STATISTICS, n_bootstrap, seed)
        row: dict[str, Any] = {"model": model_name, "n_test": len(y_test)}
        for name, function in TEST_STATISTICS.items():
            low, high = compute_percentile_interval(replicates[name], confidence_level)
            row[name] = function(y_test, predicted)
            row[f"{name}_low"], row[f"{name}_high"] = low, high
        rows.append(row)
    return pl.DataFrame(rows)


def compare_models(
    cv_results: dict[str, list[FoldResult]],
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    settings: Settings,
) -> dict[str, Any]:
    """Testa se o gradient boosting difere da regressão linear (CV e teste).

    Parameters
    ----------
    cv_results : dict[str, list[FoldResult]]
        Resultados por dobra de cada modelo (mesmas dobras).
    predictions : dict[str, np.ndarray]
        Previsões do teste por modelo.
    y_test : np.ndarray
        Notas observadas no teste.
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    dict[str, Any]
        Teste t reamostrado corrigido sobre o RMSE por dobra e IC bootstrap
        pareado da diferença de RMSE (linear - boosting) no teste.
    """
    linear = cv_results[LINEAR_MODEL_NAME]
    boosting = cv_results[GRADIENT_BOOSTING_MODEL_NAME]
    differences = [a.rmse - b.rmse for a, b in zip(linear, boosting, strict=True)]
    ttest = corrected_resampled_ttest(differences, linear[0].n_train, linear[0].n_test)
    replicates = bootstrap_metric_difference(
        y_test,
        predictions[LINEAR_MODEL_NAME],
        predictions[GRADIENT_BOOSTING_MODEL_NAME],
        calculate_rmse,
        settings.modeling.n_bootstrap,
        settings.project.random_seed,
    )
    low, high = compute_percentile_interval(replicates, settings.modeling.confidence_level)
    test_difference = calculate_rmse(y_test, predictions[LINEAR_MODEL_NAME]) - calculate_rmse(
        y_test, predictions[GRADIENT_BOOSTING_MODEL_NAME]
    )
    return {
        "metric": "rmse (linear - gradient_boosting)",
        "cv_mean_difference": float(np.mean(differences)),
        "cv_corrected_ttest": {
            "statistic": ttest.statistic,
            "p_value": ttest.p_value,
            "degrees_of_freedom": ttest.degrees_of_freedom,
        },
        "test_difference": test_difference,
        "test_difference_ci": [low, high],
    }


def run_training(settings: Settings) -> dict[str, Any]:
    """Treina e avalia a regressão linear (referência) e o gradient boosting.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.

    Returns
    -------
    dict[str, Any]
        Resumo gravado em ``reports/tables/model_comparison.json``.

    Raises
    ------
    FileNotFoundError
        Se a tabela gold não existir (execute a etapa ``prepare``).

    Notes
    -----
    Fluxo: amostra → divisão treino/teste → validação cruzada K-fold no treino
    (IC por t de Student) → teste t corrigido entre modelos → reajuste no
    treino completo → métricas no teste com IC bootstrap. O teste nunca é
    usado para ajustar nada. As previsões do teste são gravadas para as etapas
    de explicabilidade e equidade.
    """
    modeling = settings.modeling
    seed = settings.project.random_seed
    tables_dir = settings.paths.resolved("tables_dir")
    data, gold_path = prepare_modeling_data(settings)
    factories = build_model_factories(settings, data.features)

    cv_results = {
        name: cross_validate_model(
            factories[name],
            data.X_train[name],
            data.y_train,
            modeling.n_splits,
            seed,
            description=f"CV {name}",
        )
        for name in MODEL_NAMES
    }
    write_table(
        pl.concat(list(starmap(results_to_frame, cv_results.items()))),
        tables_dir / "cv_folds.csv",
    )
    cv_summary = summarize_cross_validation(cv_results, modeling.confidence_level)
    write_table(cv_summary, tables_dir / "cv_summary.csv")

    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        model: Regressor = factories[name]()
        model.fit(data.X_train[name], data.y_train)
        fitted[name] = model
        predictions[name] = np.asarray(model.predict(data.X_test[name]))
        save_model(model, settings.paths.resolved("models_dir") / MODEL_FILES[name])

    test_metrics = evaluate_on_test(
        predictions, data.y_test, modeling.n_bootstrap, modeling.confidence_level, seed
    )
    write_table(test_metrics, tables_dir / "test_metrics.csv")
    write_table(
        fitted[LINEAR_MODEL_NAME].summarize_coefficients(1 - modeling.confidence_level),
        tables_dir / "linear_coefficients.csv",
    )
    write_parquet(
        data.test.with_columns(
            [pl.Series(f"{PREDICTION_PREFIX}{name}", predictions[name]) for name in MODEL_NAMES]
        ),
        settings.paths.resolved("gold_dir") / TEST_PREDICTIONS_FILE,
    )
    figures_dir = settings.paths.resolved("figures_dir")
    save_figure(plot_model_comparison(test_metrics), "model_comparison", figures_dir)

    summary = {
        "gold_sha256": hash_file(gold_path),
        "n_train": data.train.height,
        "n_test": data.test.height,
        "random_seed": seed,
        "comparison": compare_models(cv_results, predictions, data.y_test, settings),
    }
    write_json(summary, tables_dir / "model_comparison.json")
    logger.info("Treinamento concluído: %s", summary["comparison"])
    return summary
