"""Fábricas de dados sintéticos para os testes (nunca dados reais do INEP)."""

import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from constants.columns import QUESTIONNAIRE_COLUMNS, YEAR_COLUMN
from preprocessing.cleaning import transform_raw_to_silver

STATES = ["SP", "RJ", "MG", "BA", "AM", "RS", "GO", "PE"]
INCOME_LETTERS = list("ABCDEFGHIJKLMNOPQ")
PARENT_EDUCATION = list("ABCDEFGH")
GENERIC_ANSWERS = list("ABCDE")


def _scores(rng: np.random.Generator, signal: np.ndarray, offset: float, present: np.ndarray):
    """Gera notas como texto (nulas para quem faltou), crescentes com a renda."""
    values = 350.0 + offset + signal + rng.normal(0.0, 30.0, len(signal))
    return [f"{value:.1f}" if here else None for value, here in zip(values, present, strict=True)]


def build_raw_frame(n_rows: int = 600, year: int = 2023, seed: int = 0) -> pl.DataFrame:
    """Cria um DataFrame no formato do CSV bruto do INEP (tudo texto).

    Parameters
    ----------
    n_rows : int, optional
        Número de inscritos sintéticos.
    year : int, optional
        Ano da edição.
    seed : int, optional
        Semente do gerador.

    Returns
    -------
    pl.DataFrame
        Tabela com todas as colunas usadas, mais colunas que o pipeline deve
        ignorar (identificador e texto acentuado). A nota cresce com a renda.
    """
    rng = np.random.default_rng(seed)
    income = rng.integers(0, len(INCOME_LETTERS), n_rows)
    signal = 12.0 * income
    present = rng.random(n_rows) > 0.1
    presence = np.where(present, "1", "0").tolist()
    status = [("3" if rng.random() < 0.03 else "1") if here else None for here in present.tolist()]

    data: dict[str, list] = {
        "NU_INSCRICAO": [str(1_000_000 + index) for index in range(n_rows)],
        "NU_ANO": [str(year)] * n_rows,
        "IN_TREINEIRO": rng.choice(["0", "1"], size=n_rows, p=[0.95, 0.05]).tolist(),
        "TP_COR_RACA": rng.choice(
            ["0", "1", "2", "3", "4", "5"], size=n_rows, p=[0.02, 0.45, 0.1, 0.38, 0.03, 0.02]
        ).tolist(),
        "TP_ESCOLA": rng.choice(["1", "2", "3"], size=n_rows).tolist(),
        "SG_UF_PROVA": rng.choice(STATES, size=n_rows).tolist(),
        "TP_PRESENCA_CN": presence,
        "TP_PRESENCA_CH": presence,
        "TP_PRESENCA_LC": presence,
        "TP_PRESENCA_MT": presence,
        "TP_STATUS_REDACAO": status,
        "NU_NOTA_CN": _scores(rng, signal, 0.0, present),
        "NU_NOTA_CH": _scores(rng, signal, 20.0, present),
        "NU_NOTA_LC": _scores(rng, signal, 10.0, present),
        "NU_NOTA_MT": _scores(rng, signal, 40.0, present),
        "NU_NOTA_REDACAO": _scores(rng, signal, -30.0, present),
        "NO_MUNICIPIO_PROVA": ["São Paulo"] * n_rows,  # acentuado: testa o latin-1
    }
    for column in QUESTIONNAIRE_COLUMNS:
        if column == "Q006":
            data[column] = [INCOME_LETTERS[index] for index in income]
        elif column == "Q005":
            data[column] = [str(value) for value in rng.integers(1, 9, n_rows)]
        elif column in {"Q001", "Q002", "Q003", "Q004"}:
            data[column] = rng.choice(PARENT_EDUCATION, size=n_rows).tolist()
        else:
            data[column] = rng.choice(GENERIC_ANSWERS, size=n_rows).tolist()
    data["Q007"] = [None if rng.random() < 0.03 else value for value in data["Q007"]]
    return pl.DataFrame(data, schema=dict.fromkeys(data, pl.String))


def build_silver_lazy(n_rows: int = 600, year: int = 2023, seed: int = 0) -> pl.LazyFrame:
    """Cria o plano lazy silver (tipado) com a coluna ``year`` já derivada."""
    raw = build_raw_frame(n_rows, year, seed).lazy()
    return transform_raw_to_silver(raw).with_columns(pl.col("NU_ANO").alias(YEAR_COLUMN))


def write_raw_csv(frame: pl.DataFrame, path: Path) -> Path:
    """Grava o DataFrame como CSV ``;`` em latin-1, como o arquivo do INEP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.write_csv(separator=";")
    path.write_bytes(text.encode("latin-1"))
    return path


def write_microdata_zip(zip_path: Path, frame: pl.DataFrame, year: int) -> Path:
    """Cria um ZIP no layout do INEP (``DADOS/MICRODADOS_ENEM_<ano>.csv``)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.write_csv(separator=";")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"DADOS/MICRODADOS_ENEM_{year}.csv", text.encode("latin-1"))
        archive.writestr("LEIA-ME/dicionario.txt", b"dicionario")
    return zip_path
