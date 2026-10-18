"""Valores SHAP do gradient boosting: importância global e por grupo."""

import logging
from collections.abc import Sequence

import numpy as np
import polars as pl
import xgboost as xgb
from xgboost import XGBRegressor

from constants.labels import QUESTION_LABELS

logger = logging.getLogger(__name__)


def compute_shap_values(model: XGBRegressor, X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Calcula os valores SHAP exatos (TreeSHAP) de um modelo de árvores.

    Parameters
    ----------
    model : XGBRegressor
        Modelo treinado.
    X : np.ndarray
        Matriz de features ``(n, p)`` (mesma codificação do treino).

    Returns
    -------
    np.ndarray
        Matriz ``(n, p)``; cada valor é a contribuição da feature, em pontos
        de nota, para afastar a previsão da média do modelo.

    Examples
    --------
    >>> values = compute_shap_values(model, X_sample)  # doctest: +SKIP
    """
    # `pred_contribs` é o TreeSHAP nativo do XGBoost. Evita o parser de `base_score`
    # do `shap.TreeExplainer`, que quebra em versões antigas do shap (Python 3.10)
    # com xgboost >= 3.1 (`could not convert string to float: '[7.1E1]'`).
    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    values = np.asarray(contributions)[:, :-1]  # a última coluna é o viés (base value)
    logger.info("SHAP calculado para %d linhas x %d features", *values.shape)
    return values


def summarize_global_importance(
    shap_values: np.ndarray, feature_names: Sequence[str]
) -> pl.DataFrame:
    """Importância global: média do valor absoluto do SHAP por feature.

    Parameters
    ----------
    shap_values : np.ndarray
        Matriz ``(n, p)`` de valores SHAP.
    feature_names : Sequence[str]
        Nomes das ``p`` features.

    Returns
    -------
    pl.DataFrame
        Colunas ``feature``, ``label``, ``mean_abs_shap`` e ``mean_shap``,
        ordenadas pela importância (decrescente).
    """
    return pl.DataFrame(
        {
            "feature": list(feature_names),
            "label": [QUESTION_LABELS.get(name, name) for name in feature_names],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort("mean_abs_shap", descending=True)


def summarize_shap_by_group(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    groups: np.ndarray,
    attribute: str,
    min_group_size: int,
) -> pl.DataFrame:
    """Importância SHAP calculada separadamente para cada grupo.

    Parameters
    ----------
    shap_values : np.ndarray
        Matriz ``(n, p)`` de valores SHAP.
    feature_names : Sequence[str]
        Nomes das ``p`` features.
    groups : np.ndarray
        Grupo de cada uma das ``n`` linhas.
    attribute : str
        Nome do atributo de agrupamento (por exemplo, ``"income_group"``).
    min_group_size : int
        Grupos menores que isto são omitidos.

    Returns
    -------
    pl.DataFrame
        Formato longo: ``attribute``, ``group``, ``n``, ``feature``, ``label``,
        ``mean_abs_shap`` e ``mean_shap``.
    """
    frames: list[pl.DataFrame] = []
    for group in np.unique(groups):
        mask = groups == group
        n_rows = int(mask.sum())
        if n_rows < min_group_size:
            logger.warning("Grupo '%s' (%s) omitido no SHAP: n=%d", group, attribute, n_rows)
            continue
        summary = summarize_global_importance(shap_values[mask], feature_names).with_columns(
            pl.lit(attribute).alias("attribute"),
            pl.lit(str(group)).alias("group"),
            pl.lit(n_rows).alias("n"),
        )
        leading = ["attribute", "group", "n"]
        frames.append(summary.select(*leading, pl.exclude(*leading)))
    return pl.concat(frames) if frames else pl.DataFrame()
