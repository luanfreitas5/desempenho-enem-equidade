"""Leitura lazy do CSV bruto dos microdados."""

from pathlib import Path
from typing import Literal

import polars as pl


def scan_raw_csv(
    csv_path: Path,
    separator: str = ";",
    encoding: Literal["utf8", "utf8-lossy"] = "utf8-lossy",
) -> pl.LazyFrame:
    """Abre o CSV bruto do INEP de forma lazy, com todas as colunas como texto.

    Parameters
    ----------
    csv_path : Path
        CSV de microdados (separador ``;``).
    separator : str, optional
        Separador de campos, by default ";".
    encoding : {"utf8", "utf8-lossy"}, optional
        Codificação aceita pelo ``scan_csv`` lazy, by default "utf8-lossy".
        O arquivo do INEP é latin-1; o modo *lossy* só substitui bytes
        acentuados, que ficam em colunas de texto que o projeto não usa.

    Returns
    -------
    pl.LazyFrame
        Plano lazy; nada é lido até ``collect``/``sink_*``.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Notes
    -----
    ``infer_schema=False`` mantém tudo como texto: a tipagem é feita
    explicitamente na transformação raw → silver, evitando inferências erradas.

    Examples
    --------
    >>> lazy = scan_raw_csv(Path("MICRODADOS_ENEM_2023.csv"))  # doctest: +SKIP
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV de microdados não encontrado: {csv_path}")
    return pl.scan_csv(csv_path, separator=separator, encoding=encoding, infer_schema=False)
