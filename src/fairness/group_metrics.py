"""Métricas de desempenho e de erro por grupo, com IC bootstrap e diferenças vs. referência."""

import logging

import numpy as np
import polars as pl

from evaluation.metrics import (
    MetricFunction,
    bootstrap_statistics,
    calculate_mae,
    calculate_mean_error,
    calculate_mean_target,
    calculate_r2,
    calculate_rmse,
    compute_percentile_interval,
)

logger = logging.getLogger(__name__)

GROUP_STATISTICS: dict[str, MetricFunction] = {
    "mean_score": calculate_mean_target,
    "mean_error": calculate_mean_error,
    "mae": calculate_mae,
    "rmse": calculate_rmse,
    "r2": calculate_r2,
}
GAP_STATISTICS: tuple[str, ...] = ("mean_score", "mean_error", "mae", "rmse")
GAP_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "attribute": pl.String,
    "group": pl.String,
    "reference": pl.String,
    "statistic": pl.String,
    "gap": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "significant": pl.Boolean,
}


def _summarize_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Estimativas pontuais e réplicas bootstrap de todas as estatísticas de um grupo."""
    estimates = {name: function(y_true, y_pred) for name, function in GROUP_STATISTICS.items()}
    replicates = bootstrap_statistics(y_true, y_pred, GROUP_STATISTICS, n_bootstrap, seed)
    row: dict[str, float] = {}
    for name, value in estimates.items():
        low, high = compute_percentile_interval(replicates[name], confidence_level)
        row.update({name: value, f"{name}_low": low, f"{name}_high": high})
    return row, replicates


def _build_gap_rows(
    attribute: str,
    group: str,
    reference: str,
    estimates: dict[str, dict[str, float]],
    replicates: dict[str, dict[str, np.ndarray]],
    confidence_level: float,
) -> list[dict]:
    """Diferenças ``grupo - referência`` com IC (réplicas independentes por grupo)."""
    rows: list[dict] = []
    for statistic in GAP_STATISTICS:
        gap = estimates[group][statistic] - estimates[reference][statistic]
        samples = replicates[group][statistic] - replicates[reference][statistic]
        low, high = compute_percentile_interval(samples, confidence_level)
        rows.append(
            {
                "attribute": attribute,
                "group": group,
                "reference": reference,
                "statistic": statistic,
                "gap": gap,
                "ci_low": low,
                "ci_high": high,
                "significant": not (low <= 0.0 <= high),
            }
        )
    return rows


def analyze_group_equity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    attribute: str,
    reference_group: str,
    n_bootstrap: int,
    confidence_level: float,
    min_group_size: int,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compara desempenho e erro do modelo entre grupos, com intervalos de confiança.

    Parameters
    ----------
    y_true : np.ndarray
        Notas observadas.
    y_pred : np.ndarray
        Notas previstas (conjunto de teste).
    groups : np.ndarray
        Grupo de cada linha (por exemplo, a faixa de renda).
    attribute : str
        Nome do atributo analisado (aparece nas tabelas).
    reference_group : str
        Grupo de referência das diferenças.
    n_bootstrap : int
        Número de réplicas bootstrap.
    confidence_level : float
        Nível de confiança dos intervalos.
    min_group_size : int
        Grupos menores que isto são omitidos (IC instável).
    seed : int
        Semente base do bootstrap.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        ``(metrics, gaps)``. ``metrics`` tem uma linha por grupo com ``n``,
        ``mean_score``, ``mean_error`` (viés), ``mae``, ``rmse`` e ``r2``,
        cada um com ``_low``/``_high``. ``gaps`` traz as diferenças
        ``grupo - referência`` em nota média, viés, MAE e RMSE, com IC e o
        indicador ``significant`` (IC exclui zero).

    Notes
    -----
    Os IC são do tipo percentil sobre reamostragem das linhas; as réplicas de
    grupos distintos são independentes, então a diferença das réplicas gera o
    IC da diferença. O R² por grupo não é comparável entre grupos com
    variâncias diferentes de nota; prefira MAE/RMSE e o viés.

    Examples
    --------
    >>> metrics, gaps = analyze_group_equity(  # doctest: +SKIP
    ...     y, pred, income, "income_group", "Mais de 6 SM", 500, 0.95, 100, 42
    ... )
    """
    estimates: dict[str, dict[str, float]] = {}
    replicates: dict[str, dict[str, np.ndarray]] = {}
    sizes: dict[str, int] = {}
    labels, counts = np.unique(groups, return_counts=True)
    for offset, (label, count) in enumerate(zip(labels, counts, strict=True)):
        if count < min_group_size:
            logger.warning(
                "Grupo '%s' (%s) omitido: n=%d < %d", label, attribute, count, min_group_size
            )
            continue
        mask = groups == label
        estimates[str(label)], replicates[str(label)] = _summarize_group(
            y_true[mask], y_pred[mask], n_bootstrap, confidence_level, seed + offset
        )
        sizes[str(label)] = int(count)

    metrics = pl.DataFrame(
        [
            {"attribute": attribute, "group": name, "n": sizes[name]} | estimates[name]
            for name in estimates
        ]
    )
    gap_rows: list[dict] = []
    if reference_group in estimates:
        for name in estimates:
            if name != reference_group:
                gap_rows.extend(
                    _build_gap_rows(
                        attribute, name, reference_group, estimates, replicates, confidence_level
                    )
                )
    else:
        logger.warning("Grupo de referência '%s' ausente em '%s'", reference_group, attribute)
    return metrics, pl.DataFrame(gap_rows, schema=GAP_SCHEMA)
