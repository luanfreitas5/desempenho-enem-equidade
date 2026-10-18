"""Exceções relacionadas a dados."""

from exceptions.base import EnemEquityError


class DataDownloadError(EnemEquityError):
    """Falha ao baixar ou extrair os microdados do INEP."""


class MissingColumnsError(EnemEquityError):
    """Colunas obrigatórias ausentes no conjunto de dados."""


class DataValidationError(EnemEquityError):
    """Dados que violam o contrato (esquema) da camada."""


class InsufficientDataError(EnemEquityError):
    """Quantidade de dados insuficiente para a etapa solicitada."""
