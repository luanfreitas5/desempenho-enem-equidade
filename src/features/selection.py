"""Seleção de features do modelo e proteção contra vazamento do alvo."""

import logging
from collections.abc import Iterable

from constants.columns import (
    LEAKAGE_PREFIXES,
    QUESTIONNAIRE_COLUMNS,
    SENSITIVE_COLUMNS,
    TARGET_COLUMN,
)
from exceptions.model import LeakageError

logger = logging.getLogger(__name__)


def validate_no_leakage(feature_names: Iterable[str]) -> None:
    """Falha se alguma feature vazar informação da nota (alvo).

    Parameters
    ----------
    feature_names : Iterable[str]
        Nomes das colunas candidatas a feature.

    Raises
    ------
    LeakageError
        Se houver o próprio alvo, notas, gabaritos, presença/status de prova
        ou qualquer coluna com prefixo listado em ``LEAKAGE_PREFIXES``.

    Examples
    --------
    >>> validate_no_leakage(["Q001", "Q006"])
    """
    offenders = sorted(
        name for name in feature_names if name == TARGET_COLUMN or name.startswith(LEAKAGE_PREFIXES)
    )
    if offenders:
        logger.error("Vazamento de alvo detectado nas features: %s", offenders)
        raise LeakageError(f"Features que vazam a nota: {offenders}")


def select_feature_columns() -> list[str]:
    """Devolve as features do modelo: apenas o questionário socioeconômico.

    Raça/cor, tipo de escola e região **não** entram como features: são usados
    apenas para agrupar a análise de equidade. A renda (``Q006``) faz parte do
    questionário e, portanto, do perfil socioeconômico modelado.

    Returns
    -------
    list[str]
        ``Q001`` a ``Q025``.

    Raises
    ------
    LeakageError
        Se alguma feature vazar o alvo.
    """
    features = list(QUESTIONNAIRE_COLUMNS)
    validate_no_leakage(features)
    overlap = sorted(set(features) & set(SENSITIVE_COLUMNS))
    if overlap:
        raise LeakageError(f"Atributos de equidade não podem ser features: {overlap}")
    return features
