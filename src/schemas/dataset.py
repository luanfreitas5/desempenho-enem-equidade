"""Contratos de dados das camadas silver e gold (Pandera, backend Polars)."""

import logging

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from constants.columns import (
    ESSAY_SCORE_COLUMN,
    ESSAY_STATUS_COLUMN,
    INCOME_GROUP_COLUMN,
    PRESENCE_COLUMNS,
    QUESTIONNAIRE_COLUMNS,
    RACE_COLUMN,
    RACE_GROUP_COLUMN,
    RAW_YEAR_COLUMN,
    REGION_COLUMN,
    SCHOOL_GROUP_COLUMN,
    SCHOOL_TYPE_COLUMN,
    SCORE_COLUMNS,
    STATE_COLUMN,
    TARGET_COLUMN,
    TRAINEE_COLUMN,
    YEAR_COLUMN,
)
from constants.labels import (
    INCOME_GROUP_ORDER,
    MISSING_CATEGORY,
    RACE_LABELS,
    REGION_BY_STATE,
    SCHOOL_TYPE_LABELS,
    UNKNOWN_GROUP,
)
from exceptions.data import DataValidationError

logger = logging.getLogger(__name__)

QUESTION_PATTERN = r"^([A-Q]|\d{1,2})$"
GOLD_QUESTION_PATTERN = rf"^([A-Q]|\d{{1,2}}|{MISSING_CATEGORY})$"
SCORE_RANGE = (0.0, 1000.0)


def build_silver_schema() -> pa.DataFrameSchema:
    """Constrói o contrato da camada silver (dados tipados, ainda sem filtros).

    Returns
    -------
    pa.DataFrameSchema
        Esquema estrito: tipos, domínios dos códigos do INEP e faixa das notas.
        Nulos são permitidos (ausentes e faltosos são tratados na camada gold).
    """
    columns: dict[str, pa.Column] = {
        RAW_YEAR_COLUMN: pa.Column(pl.Int16, pa.Check.in_range(2009, 2100)),
        TRAINEE_COLUMN: pa.Column(pl.Int8, pa.Check.isin([0, 1]), nullable=True),
        RACE_COLUMN: pa.Column(pl.Int8, pa.Check.isin(list(RACE_LABELS)), nullable=True),
        SCHOOL_TYPE_COLUMN: pa.Column(
            pl.Int8, pa.Check.isin(list(SCHOOL_TYPE_LABELS)), nullable=True
        ),
        STATE_COLUMN: pa.Column(pl.String, pa.Check.isin(list(REGION_BY_STATE)), nullable=True),
        ESSAY_STATUS_COLUMN: pa.Column(pl.Int8, pa.Check.in_range(1, 9), nullable=True),
    }
    for name in PRESENCE_COLUMNS:
        columns[name] = pa.Column(pl.Int8, pa.Check.isin([0, 1, 2]), nullable=True)
    for name in (*SCORE_COLUMNS, ESSAY_SCORE_COLUMN):
        columns[name] = pa.Column(pl.Float64, pa.Check.in_range(*SCORE_RANGE), nullable=True)
    for name in QUESTIONNAIRE_COLUMNS:
        columns[name] = pa.Column(pl.String, pa.Check.str_matches(QUESTION_PATTERN), nullable=True)
    return pa.DataFrameSchema(columns, strict=True, name="silver_enem")


def build_gold_schema() -> pa.DataFrameSchema:
    """Constrói o contrato da camada gold (tabela analítica sem nulos).

    Returns
    -------
    pa.DataFrameSchema
        Esquema estrito: alvo em [0, 1000], grupos de equidade com domínio
        conhecido e questionário sem nulos (ausentes viram ``ND``).
    """
    columns: dict[str, pa.Column] = {
        YEAR_COLUMN: pa.Column(pl.Int16, pa.Check.in_range(2009, 2100)),
        TARGET_COLUMN: pa.Column(pl.Float64, pa.Check.in_range(*SCORE_RANGE)),
        INCOME_GROUP_COLUMN: pa.Column(
            pl.String, pa.Check.isin([*INCOME_GROUP_ORDER, UNKNOWN_GROUP])
        ),
        RACE_GROUP_COLUMN: pa.Column(
            pl.String, pa.Check.isin([*RACE_LABELS.values(), UNKNOWN_GROUP])
        ),
        SCHOOL_GROUP_COLUMN: pa.Column(
            pl.String, pa.Check.isin([*SCHOOL_TYPE_LABELS.values(), UNKNOWN_GROUP])
        ),
        REGION_COLUMN: pa.Column(
            pl.String, pa.Check.isin([*set(REGION_BY_STATE.values()), UNKNOWN_GROUP])
        ),
    }
    for name in QUESTIONNAIRE_COLUMNS:
        columns[name] = pa.Column(pl.String, pa.Check.str_matches(GOLD_QUESTION_PATTERN))
    return pa.DataFrameSchema(columns, strict=True, name="gold_enem")


def _validate(frame: pl.DataFrame, schema: pa.DataFrameSchema) -> pl.DataFrame:
    """Valida o DataFrame e converte falhas do Pandera em ``DataValidationError``."""
    try:
        return schema.validate(frame)
    except (SchemaError, SchemaErrors) as error:
        logger.exception("Violação do contrato '%s'", schema.name)
        raise DataValidationError(f"Dados violam o contrato '{schema.name}': {error}") from error


def validate_silver(frame: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame contra o contrato silver.

    Parameters
    ----------
    frame : pl.DataFrame
        Dados da camada silver (colunas ``INGEST_COLUMNS``).

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, se válido.

    Raises
    ------
    DataValidationError
        Se o contrato for violado.
    """
    return _validate(frame, build_silver_schema())


def validate_gold(frame: pl.DataFrame) -> pl.DataFrame:
    """Valida um DataFrame contra o contrato gold.

    Parameters
    ----------
    frame : pl.DataFrame
        Dados da camada gold.

    Returns
    -------
    pl.DataFrame
        O mesmo DataFrame, se válido.

    Raises
    ------
    DataValidationError
        Se o contrato for violado.
    """
    return _validate(frame, build_gold_schema())
