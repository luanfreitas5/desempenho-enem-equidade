"""Regras de presença/validade dos participantes e funil de seleção."""

import logging

import polars as pl

from constants.columns import (
    ESSAY_SCORE_COLUMN,
    ESSAY_STATUS_COLUMN,
    PRESENCE_COLUMNS,
    SCORE_COLUMNS,
    TRAINEE_COLUMN,
)

logger = logging.getLogger(__name__)

FilterStep = tuple[str, pl.Expr]

PRESENT_CODE = 1  # TP_PRESENCA_*: 1 = presente (0 = faltou, 2 = eliminado)
VALID_ESSAY_STATUS = 1  # TP_STATUS_REDACAO: 1 = sem problemas


def build_filter_steps(exclude_trainees: bool = True) -> list[FilterStep]:
    """Monta as regras (rótulo, expressão) que definem os participantes válidos.

    Parameters
    ----------
    exclude_trainees : bool, optional
        Se ``True``, remove treineiros, by default True.

    Returns
    -------
    list[tuple[str, pl.Expr]]
        Regras aplicadas em sequência. Valores nulos reprovam a regra.

    Examples
    --------
    >>> [label for label, _ in build_filter_steps(False)][0]
    'Presente nas quatro provas objetivas'
    """
    steps: list[FilterStep] = []
    if exclude_trainees:
        steps.append(("Não treineiro", pl.col(TRAINEE_COLUMN) == 0))
    steps.extend(
        [
            (
                "Presente nas quatro provas objetivas",
                pl.all_horizontal([pl.col(name) == PRESENT_CODE for name in PRESENCE_COLUMNS]),
            ),
            ("Redação sem problemas", pl.col(ESSAY_STATUS_COLUMN) == VALID_ESSAY_STATUS),
            (
                "Todas as notas informadas",
                pl.all_horizontal(
                    [pl.col(name).is_not_null() for name in (*SCORE_COLUMNS, ESSAY_SCORE_COLUMN)]
                ),
            ),
        ]
    )
    return steps


def apply_filter_steps(lazy_frame: pl.LazyFrame, steps: list[FilterStep]) -> pl.LazyFrame:
    """Aplica as regras de seleção em sequência.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        Plano lazy da camada silver.
    steps : list[tuple[str, pl.Expr]]
        Regras geradas por ``build_filter_steps``.

    Returns
    -------
    pl.LazyFrame
        Plano lazy apenas com os participantes que satisfazem todas as regras.
    """
    for _, expression in steps:
        lazy_frame = lazy_frame.filter(expression)
    return lazy_frame


def compute_participation_funnel(lazy_frame: pl.LazyFrame, steps: list[FilterStep]) -> pl.DataFrame:
    """Conta quantos participantes sobram após cada regra (funil de seleção).

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        Plano lazy da camada silver.
    steps : list[tuple[str, pl.Expr]]
        Regras geradas por ``build_filter_steps``.

    Returns
    -------
    pl.DataFrame
        Colunas ``step``, ``rows``, ``dropped`` e ``retained_pct`` (em relação
        ao total de inscritos).
    """
    labels = ["Inscritos (silver)"]
    counts = [int(lazy_frame.select(pl.len()).collect().item())]
    current = lazy_frame
    for label, expression in steps:
        current = current.filter(expression)
        labels.append(label)
        counts.append(int(current.select(pl.len()).collect().item()))
    total = counts[0] or 1
    funnel = pl.DataFrame({"step": labels, "rows": counts}).with_columns(
        (pl.col("rows").shift(1) - pl.col("rows")).fill_null(0).alias("dropped"),
        (pl.col("rows") / total * 100).round(2).alias("retained_pct"),
    )
    logger.info("Funil de seleção:\n%s", funnel)
    return funnel
