"""Codificação das respostas do questionário para os modelos."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from constants.columns import NUMERIC_QUESTIONS

LETTER_CODES: dict[str, int] = {chr(ord("A") + offset): offset for offset in range(26)}


def build_ordinal_expressions(columns: Sequence[str]) -> list[pl.Expr]:
    """Cria as expressões que convertem respostas (A, B, C ...) em códigos ordinais.

    Parameters
    ----------
    columns : Sequence[str]
        Colunas do questionário.

    Returns
    -------
    list[pl.Expr]
        Uma expressão ``Float32`` por coluna. Letras viram 0, 1, 2 ...; questões
        numéricas (``Q005``) viram o próprio número; a categoria ``ND``
        (ausente) vira nulo.
    """
    expressions: list[pl.Expr] = []
    for column in columns:
        if column in NUMERIC_QUESTIONS:
            expression = pl.col(column).cast(pl.Float32, strict=False)
        else:
            expression = pl.col(column).replace_strict(
                LETTER_CODES, default=None, return_dtype=pl.Float32
            )
        expressions.append(expression.alias(column))
    return expressions


def encode_ordinal(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Codifica o questionário em uma matriz ordinal (ausentes = ``NaN``).

    Adequada para gradient boosting, que lida nativamente com ``NaN`` e com
    relações não lineares na ordem das respostas.

    Parameters
    ----------
    frame : pl.DataFrame
        Tabela com as colunas do questionário.
    columns : Sequence[str]
        Colunas a codificar (define a ordem das colunas da matriz).

    Returns
    -------
    np.ndarray
        Matriz ``float32`` de formato ``(n_linhas, n_colunas)``.

    Examples
    --------
    >>> encode_ordinal(pl.DataFrame({"Q001": ["A", "C", "ND"]}), ["Q001"]).ravel()[:2]
    array([0., 2.], dtype=float32)
    """
    return (
        frame.select(build_ordinal_expressions(columns)).to_numpy().astype(np.float32, copy=False)
    )


def encode_categorical(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Extrai as respostas como matriz de texto, para o one-hot da regressão linear.

    Parameters
    ----------
    frame : pl.DataFrame
        Tabela com as colunas do questionário.
    columns : Sequence[str]
        Colunas a extrair.

    Returns
    -------
    np.ndarray
        Matriz de texto (``object``) de formato ``(n_linhas, n_colunas)``.
    """
    return frame.select(list(columns)).to_numpy().astype(object)
