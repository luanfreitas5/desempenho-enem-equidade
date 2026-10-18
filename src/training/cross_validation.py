"""Validação cruzada K-fold com métricas por dobra."""

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from sklearn.model_selection import KFold

from evaluation.metrics import compute_regression_metrics
from models.base import Regressor
from utils.progress import create_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldResult:
    """Métricas de uma dobra da validação cruzada."""

    fold: int
    n_train: int
    n_test: int
    rmse: float
    mae: float
    r2: float


def cross_validate_model(
    model_factory: Callable[[], Regressor],
    X: np.ndarray,  # noqa: N803 - convenção sklearn
    y: np.ndarray,
    n_splits: int,
    seed: int,
    description: str = "Validação cruzada",
) -> list[FoldResult]:
    """Executa K-fold embaralhado, treinando um modelo novo em cada dobra.

    Parameters
    ----------
    model_factory : Callable[[], Regressor]
        Função sem argumentos que devolve um modelo não treinado.
    X : np.ndarray
        Matriz de features ``(n, p)``.
    y : np.ndarray
        Alvo ``(n,)``.
    n_splits : int
        Número de dobras.
    seed : int
        Semente do embaralhamento.
    description : str, optional
        Texto da barra de progresso, by default "Validação cruzada".

    Returns
    -------
    list[FoldResult]
        Uma entrada por dobra, com RMSE, MAE e R² no conjunto de validação.

    Notes
    -----
    Cada dobra treina somente com suas linhas de treino: nenhuma estatística é
    calculada com as linhas de validação.

    Examples
    --------
    >>> results = cross_validate_model(factory, X, y, 5, seed=42)  # doctest: +SKIP
    """
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: list[FoldResult] = []
    with create_progress() as progress:
        task = progress.add_task(description, total=n_splits)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            predictions = np.asarray(model.predict(X[valid_idx]))
            metrics = compute_regression_metrics(y[valid_idx], predictions)
            results.append(
                FoldResult(fold=fold, n_train=len(train_idx), n_test=len(valid_idx), **metrics)
            )
            logger.info("%s — dobra %d: RMSE=%.2f", description, fold, metrics["rmse"])
            progress.advance(task)
    return results


def results_to_frame(model_name: str, results: list[FoldResult]) -> pl.DataFrame:
    """Converte os resultados por dobra em tabela, identificando o modelo.

    Parameters
    ----------
    model_name : str
        Nome do modelo.
    results : list[FoldResult]
        Resultados de ``cross_validate_model``.

    Returns
    -------
    pl.DataFrame
        Uma linha por dobra, com a coluna ``model``.
    """
    return pl.DataFrame([asdict(result) for result in results]).with_columns(
        pl.lit(model_name).alias("model")
    )
