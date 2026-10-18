"""Divisão treino/teste reprodutível."""

import numpy as np
import polars as pl

from exceptions.data import InsufficientDataError


def split_train_test(
    frame: pl.DataFrame, test_size: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide a tabela em treino e teste por permutação aleatória das linhas.

    Parameters
    ----------
    frame : pl.DataFrame
        Tabela completa.
    test_size : float
        Fração das linhas reservada ao teste (entre 0 e 1, exclusivo).
    seed : int
        Semente da permutação.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(treino, teste)``, disjuntos.

    Raises
    ------
    InsufficientDataError
        Se algum dos conjuntos ficar vazio.

    Examples
    --------
    >>> train, test = split_train_test(pl.DataFrame({"a": range(10)}), 0.2, seed=0)
    >>> (train.height, test.height)
    (8, 2)
    """
    n_rows = frame.height
    n_test = round(n_rows * test_size)
    if n_test < 1 or n_rows - n_test < 1:
        raise InsufficientDataError(
            f"Dados insuficientes ({n_rows} linhas) para test_size={test_size}"
        )
    permutation = np.random.default_rng(seed).permutation(n_rows)
    test_idx = np.sort(permutation[:n_test])
    train_idx = np.sort(permutation[n_test:])
    return frame[train_idx], frame[test_idx]
