"""Amostragem reprodutível de linhas."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def sample_rows(frame: pl.DataFrame, sample_size: int | None, seed: int) -> pl.DataFrame:
    """Sorteia ``sample_size`` linhas sem reposição, com semente fixa.

    Parameters
    ----------
    frame : pl.DataFrame
        Tabela de origem.
    sample_size : int or None
        Tamanho da amostra; ``None`` ou maior que a tabela devolve tudo.
    seed : int
        Semente do sorteio.

    Returns
    -------
    pl.DataFrame
        Amostra (ou a própria tabela, se não for necessário amostrar).

    Examples
    --------
    >>> sample_rows(pl.DataFrame({"a": range(10)}), 3, seed=1).height
    3
    """
    if sample_size is None or sample_size >= frame.height:
        return frame
    logger.info("Amostrando %d de %d linhas (semente %d)", sample_size, frame.height, seed)
    return frame.sample(n=sample_size, seed=seed)
