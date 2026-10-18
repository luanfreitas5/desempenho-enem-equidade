"""Exceções relacionadas a configuração."""

from exceptions.base import EnemEquityError


class ConfigurationError(EnemEquityError):
    """Arquivo de configuração ausente, malformado ou inválido."""
