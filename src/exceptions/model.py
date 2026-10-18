"""Exceções relacionadas a modelagem."""

from exceptions.base import EnemEquityError


class LeakageError(EnemEquityError):
    """Feature que vaza informação do alvo (nota) foi detectada."""


class ModelNotFittedError(EnemEquityError):
    """Modelo usado antes de ser treinado."""
