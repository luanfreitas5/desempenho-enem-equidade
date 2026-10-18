"""Testes de limpeza, filtros, engenharia de variáveis, codificação e contratos."""

import numpy as np
import polars as pl
import pytest

from constants.columns import (
    ESSAY_SCORE_COLUMN,
    GOLD_COLUMNS,
    QUESTIONNAIRE_COLUMNS,
    SCORE_COLUMNS,
    TARGET_COLUMN,
)
from constants.labels import INCOME_GROUP_BY_LETTER, MISSING_CATEGORY, REGION_BY_STATE
from exceptions.data import DataValidationError, MissingColumnsError
from exceptions.model import LeakageError
from factories import build_raw_frame, build_silver_lazy
from features.engineering import add_group_columns, build_gold_frame
from features.selection import select_feature_columns, validate_no_leakage
from preprocessing.cleaning import transform_raw_to_silver, validate_required_columns
from preprocessing.encoding import encode_categorical, encode_ordinal
from preprocessing.filtering import (
    apply_filter_steps,
    build_filter_steps,
    compute_participation_funnel,
)
from schemas.dataset import validate_gold, validate_silver


def test_constants_are_consistent():
    """Mapeamentos cobrem todas as letras de renda, UFs e questões."""
    assert set(INCOME_GROUP_BY_LETTER) == set("ABCDEFGHIJKLMNOPQ")
    assert len(REGION_BY_STATE) == 27
    assert len(QUESTIONNAIRE_COLUMNS) == 25
    assert TARGET_COLUMN in GOLD_COLUMNS


def test_validate_required_columns():
    """Colunas ausentes geram MissingColumnsError listando o que falta."""
    validate_required_columns(["a", "b"], ["a"])
    with pytest.raises(MissingColumnsError, match="'c'"):
        validate_required_columns(["a"], ["a", "c"])


def test_transform_raw_to_silver_types_and_privacy(raw_frame: pl.DataFrame):
    """Tipagem explícita, texto normalizado e sem identificador de inscrição."""
    silver = transform_raw_to_silver(raw_frame.lazy()).collect()
    assert silver["TP_COR_RACA"].dtype == pl.Int8
    assert silver["NU_NOTA_MT"].dtype == pl.Float64
    assert silver["NU_ANO"].dtype == pl.Int16
    assert "NU_INSCRICAO" not in silver.columns
    assert "NO_MUNICIPIO_PROVA" not in silver.columns
    validate_silver(silver)


def test_transform_raw_to_silver_missing_column(raw_frame: pl.DataFrame):
    """Um CSV sem coluna obrigatória falha cedo."""
    with pytest.raises(MissingColumnsError):
        transform_raw_to_silver(raw_frame.drop("Q001").lazy())


def test_text_normalization_empty_becomes_null():
    """Texto vazio ou só com espaços vira nulo e letras ficam em maiúsculas."""
    raw = build_raw_frame(3).with_columns(pl.Series("Q001", [" b ", "", None], dtype=pl.String))
    silver = transform_raw_to_silver(raw.lazy()).collect()
    assert silver["Q001"].to_list() == ["B", None, None]


@pytest.mark.smoke
def test_participation_funnel_is_monotonic(silver_lazy: pl.LazyFrame):
    """O funil nunca aumenta e começa no total de inscritos."""
    steps = build_filter_steps(exclude_trainees=True)
    funnel = compute_participation_funnel(silver_lazy, steps)
    rows = funnel["rows"].to_list()
    assert rows[0] == 600
    assert rows == sorted(rows, reverse=True)
    assert funnel["dropped"][0] == 0
    assert len(build_filter_steps(exclude_trainees=False)) == len(steps) - 1


def test_gold_frame_contract_and_target(silver_lazy: pl.LazyFrame):
    """A gold tem exatamente as colunas esperadas e o alvo é a média das 5 notas."""
    filtered = apply_filter_steps(silver_lazy, build_filter_steps(True))
    gold = build_gold_frame(filtered).collect()
    reference = filtered.collect()
    assert gold.columns == list(GOLD_COLUMNS)
    expected = reference.select(pl.mean_horizontal(*SCORE_COLUMNS, ESSAY_SCORE_COLUMN).alias("e"))[
        "e"
    ].to_numpy()
    assert np.allclose(gold[TARGET_COLUMN].to_numpy(), expected)
    assert all(gold[column].null_count() == 0 for column in QUESTIONNAIRE_COLUMNS)
    assert MISSING_CATEGORY in gold["Q007"].to_list()  # ausentes viram categoria explícita
    validate_gold(gold)


def test_add_group_columns_mapping():
    """Renda, raça, escola e região são mapeadas; desconhecidos viram 'Não informado'."""
    silver = pl.LazyFrame(
        {
            "Q006": ["A", "Q", None],
            "TP_COR_RACA": [1, 5, None],
            "TP_ESCOLA": [2, 3, 1],
            "SG_UF_PROVA": ["SP", "AM", None],
        }
    )
    groups = add_group_columns(silver).collect()
    assert groups["income_group"].to_list() == ["Até 1 SM", "Mais de 6 SM", "Não informado"]
    assert groups["race"].to_list() == ["Branca", "Indígena", "Não informado"]
    assert groups["school_type"].to_list() == ["Pública", "Privada", "Não respondeu"]
    assert groups["region"].to_list() == ["Sudeste", "Norte", "Não informado"]


def test_leakage_guard():
    """Notas, presença e o próprio alvo são bloqueados como feature."""
    validate_no_leakage(["Q001", "Q006"])
    for offender in ("NU_NOTA_MT", "mean_score", "TP_PRESENCA_CN", "TP_STATUS_REDACAO"):
        with pytest.raises(LeakageError):
            validate_no_leakage(["Q001", offender])
    assert select_feature_columns() == list(QUESTIONNAIRE_COLUMNS)


def test_encode_ordinal_letters_numbers_and_missing():
    """Letras viram 0,1,2...; Q005 é numérica; 'ND' vira NaN."""
    frame = pl.DataFrame({"Q001": ["A", "C", "ND"], "Q005": ["3", "10", "ND"]})
    matrix = encode_ordinal(frame, ["Q001", "Q005"])
    assert matrix.dtype == np.float32
    assert matrix[:2].tolist() == [[0.0, 3.0], [2.0, 10.0]]
    assert np.isnan(matrix[2]).all()


def test_encode_categorical_shape():
    """A matriz categórica mantém texto e a ordem das colunas."""
    frame = pl.DataFrame({"Q001": ["A", "B"], "Q002": ["C", "D"]})
    matrix = encode_categorical(frame, ["Q002", "Q001"])
    assert matrix.tolist() == [["C", "A"], ["D", "B"]]


def test_schemas_reject_invalid_data(gold_frame: pl.DataFrame):
    """Violações do contrato viram DataValidationError."""
    with pytest.raises(DataValidationError):
        validate_gold(gold_frame.with_columns((pl.col(TARGET_COLUMN) + 2000).alias(TARGET_COLUMN)))
    with pytest.raises(DataValidationError):
        validate_gold(gold_frame.with_columns(pl.lit("X").alias("income_group")))
    with pytest.raises(DataValidationError):
        validate_gold(gold_frame.with_columns(pl.lit(1).alias("coluna_extra")))

    silver = transform_raw_to_silver(build_raw_frame(20).lazy()).collect()
    with pytest.raises(DataValidationError):
        validate_silver(silver.with_columns(pl.lit(9, dtype=pl.Int8).alias("TP_COR_RACA")))


def test_gold_not_contaminated_by_noise_columns():
    """Colunas fora do contrato nunca chegam à gold."""
    gold = build_gold_frame(
        apply_filter_steps(build_silver_lazy(100), build_filter_steps(True))
    ).collect()
    assert not any(name.startswith(("NU_NOTA", "TP_PRESENCA")) for name in gold.columns)
