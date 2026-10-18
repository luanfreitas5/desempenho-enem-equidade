"""Gravação de Parquet particionado por ano (silver) e de tabelas gold."""

import logging
from pathlib import Path

import polars as pl

from constants.columns import PARTITION_COLUMN

logger = logging.getLogger(__name__)

PART_FILE_NAME = "part-0.parquet"


def sink_year_partition(lazy_frame: pl.LazyFrame, silver_dir: Path, year: int) -> Path:
    """Grava o plano lazy como Parquet na partição ``ano=<year>`` (sem materializar tudo).

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        Plano lazy já tipado (camada silver).
    silver_dir : Path
        Pasta raiz da camada silver.
    year : int
        Ano da partição.

    Returns
    -------
    Path
        Caminho do arquivo Parquet da partição.

    Notes
    -----
    A escrita usa um arquivo temporário seguido de ``replace``: uma execução
    interrompida nunca deixa uma partição corrompida.

    Examples
    --------
    >>> sink_year_partition(lazy, Path("data/silver/enem"), 2023)  # doctest: +SKIP
    """
    partition_dir = silver_dir / f"{PARTITION_COLUMN}={year}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    target = partition_dir / PART_FILE_NAME
    temporary = target.with_suffix(".parquet.tmp")
    lazy_frame.sink_parquet(temporary, compression="zstd")
    temporary.replace(target)
    logger.info("Partição %s gravada em %s", year, target)
    return target


def write_parquet(frame: pl.DataFrame, path: Path) -> Path:
    """Grava um DataFrame como Parquet comprimido (cria a pasta se necessário).

    Parameters
    ----------
    frame : pl.DataFrame
        Tabela a gravar.
    path : Path
        Arquivo de destino.

    Returns
    -------
    Path
        O próprio ``path`` gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    return path
