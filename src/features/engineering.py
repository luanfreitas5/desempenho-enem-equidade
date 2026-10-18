"""Cria o alvo, os grupos de equidade e trata ausentes (silver → gold)."""

import polars as pl

from constants.columns import (
    ESSAY_SCORE_COLUMN,
    GOLD_COLUMNS,
    INCOME_COLUMN,
    INCOME_GROUP_COLUMN,
    QUESTIONNAIRE_COLUMNS,
    RACE_COLUMN,
    RACE_GROUP_COLUMN,
    REGION_COLUMN,
    SCHOOL_GROUP_COLUMN,
    SCHOOL_TYPE_COLUMN,
    SCORE_COLUMNS,
    STATE_COLUMN,
    TARGET_COLUMN,
)
from constants.labels import (
    INCOME_GROUP_BY_LETTER,
    MISSING_CATEGORY,
    RACE_LABELS,
    REGION_BY_STATE,
    SCHOOL_TYPE_LABELS,
    UNKNOWN_GROUP,
)


def add_target(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """Adiciona ``mean_score``: média das quatro provas objetivas e da redação.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        Plano lazy com as notas (já sem ausentes, após os filtros de presença).

    Returns
    -------
    pl.LazyFrame
        Plano com a coluna ``mean_score`` (0 a 1000).
    """
    return lazy_frame.with_columns(
        pl.mean_horizontal(*SCORE_COLUMNS, ESSAY_SCORE_COLUMN).alias(TARGET_COLUMN)
    )


def _map_group(column: str, mapping: dict, alias: str) -> pl.Expr:
    """Mapeia códigos para rótulos de grupo, com ``Não informado`` para o restante."""
    return (
        pl.col(column)
        .replace_strict(mapping, default=UNKNOWN_GROUP, return_dtype=pl.String)
        .fill_null(UNKNOWN_GROUP)
        .alias(alias)
    )


def add_group_columns(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """Deriva os grupos usados na análise de equidade.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        Plano lazy da camada silver.

    Returns
    -------
    pl.LazyFrame
        Plano com ``income_group`` (faixa de renda em salários mínimos),
        ``race`` (raça/cor), ``school_type`` (tipo de escola) e ``region``.
    """
    return lazy_frame.with_columns(
        _map_group(INCOME_COLUMN, INCOME_GROUP_BY_LETTER, INCOME_GROUP_COLUMN),
        _map_group(RACE_COLUMN, RACE_LABELS, RACE_GROUP_COLUMN),
        _map_group(SCHOOL_TYPE_COLUMN, SCHOOL_TYPE_LABELS, SCHOOL_GROUP_COLUMN),
        _map_group(STATE_COLUMN, REGION_BY_STATE, REGION_COLUMN),
    )


def impute_questionnaire(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """Trata respostas ausentes do questionário como categoria explícita ``ND``.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        Plano lazy com as colunas ``Q001`` a ``Q025``.

    Returns
    -------
    pl.LazyFrame
        Plano sem nulos no questionário. Manter a ausência como categoria
        (em vez de imputar a moda) evita inventar respostas e permite ao
        modelo aprender se "não responder" carrega informação.
    """
    return lazy_frame.with_columns(
        [pl.col(name).fill_null(MISSING_CATEGORY) for name in QUESTIONNAIRE_COLUMNS]
    )


def build_gold_frame(filtered: pl.LazyFrame) -> pl.LazyFrame:
    """Constrói a tabela analítica (gold) a partir dos participantes válidos.

    Parameters
    ----------
    filtered : pl.LazyFrame
        Plano lazy silver já filtrado por ``apply_filter_steps``.

    Returns
    -------
    pl.LazyFrame
        Plano com exatamente as colunas ``GOLD_COLUMNS``: ano, alvo, grupos de
        equidade e questionário. Notas individuais, presença e identificadores
        não são mantidos (evita vazamento e minimiza dados).
    """
    return impute_questionnaire(add_group_columns(add_target(filtered))).select(GOLD_COLUMNS)
