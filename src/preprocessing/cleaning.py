"""Transformação raw → silver: seleção de colunas, tipagem e normalização."""

import logging
from collections.abc import Iterable

import polars as pl

from constants.columns import (
    ESSAY_SCORE_COLUMN,
    ESSAY_STATUS_COLUMN,
    INGEST_COLUMNS,
    PRESENCE_COLUMNS,
    QUESTIONNAIRE_COLUMNS,
    RACE_COLUMN,
    RAW_YEAR_COLUMN,
    SCHOOL_TYPE_COLUMN,
    SCORE_COLUMNS,
    STATE_COLUMN,
    TRAINEE_COLUMN,
)
from exceptions.data import MissingColumnsError

logger = logging.getLogger(__name__)

SMALL_INT_COLUMNS: tuple[str, ...] = (
    TRAINEE_COLUMN,
    RACE_COLUMN,
    SCHOOL_TYPE_COLUMN,
    *PRESENCE_COLUMNS,
    ESSAY_STATUS_COLUMN,
)
FLOAT_COLUMNS: tuple[str, ...] = (*SCORE_COLUMNS, ESSAY_SCORE_COLUMN)
TEXT_COLUMNS: tuple[str, ...] = (STATE_COLUMN, *QUESTIONNAIRE_COLUMNS)


def validate_required_columns(available: Iterable[str], required: Iterable[str]) -> None:
    """Garante que todas as colunas obrigatórias existem.

    Parameters
    ----------
    available : Iterable[str]
        Colunas presentes no conjunto de dados.
    required : Iterable[str]
        Colunas obrigatórias.

    Raises
    ------
    MissingColumnsError
        Se alguma coluna obrigatória não estiver disponível.

    Examples
    --------
    >>> validate_required_columns(["a", "b"], ["a"])
    """
    missing = sorted(set(required) - set(available))
    if missing:
        logger.error("Colunas ausentes nos microdados: %s", missing)
        raise MissingColumnsError(f"Colunas obrigatórias ausentes: {missing}")


def _normalize_text(column: str) -> pl.Expr:
    """Remove espaços, padroniza em maiúsculas e converte texto vazio em nulo."""
    cleaned = pl.col(column).str.strip_chars().str.to_uppercase()
    return pl.when(cleaned.str.len_chars() > 0).then(cleaned).alias(column)


def transform_raw_to_silver(raw: pl.LazyFrame) -> pl.LazyFrame:
    """Seleciona, tipa e normaliza as colunas usadas, a partir do CSV bruto.

    Parameters
    ----------
    raw : pl.LazyFrame
        Plano lazy do CSV bruto (todas as colunas como texto).

    Returns
    -------
    pl.LazyFrame
        Plano lazy da camada silver: códigos como ``Int8``, notas como
        ``Float64`` e respostas do questionário como texto normalizado.
        O identificador de inscrição não é mantido (minimização/LGPD).

    Raises
    ------
    MissingColumnsError
        Se o CSV não tiver alguma das colunas necessárias.

    Examples
    --------
    >>> silver = transform_raw_to_silver(scan_raw_csv(csv_path))  # doctest: +SKIP
    """
    validate_required_columns(raw.collect_schema().names(), INGEST_COLUMNS)
    return raw.select(INGEST_COLUMNS).with_columns(
        pl.col(RAW_YEAR_COLUMN).cast(pl.Int16, strict=False),
        *[pl.col(name).cast(pl.Int8, strict=False) for name in SMALL_INT_COLUMNS],
        *[pl.col(name).cast(pl.Float64, strict=False) for name in FLOAT_COLUMNS],
        *[_normalize_text(name) for name in TEXT_COLUMNS],
    )
