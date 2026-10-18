"""Leitura lazy das camadas silver e gold."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from constants.columns import PARTITION_COLUMN, YEAR_COLUMN


def scan_silver(silver_dir: Path, years: Sequence[int] | None = None) -> pl.LazyFrame:
    """Abre as partições da camada silver como um único plano lazy.

    Parameters
    ----------
    silver_dir : Path
        Pasta raiz da camada silver (com subpastas ``ano=YYYY``).
    years : Sequence[int] or None, optional
        Anos a carregar; ``None`` carrega todas as partições.

    Returns
    -------
    pl.LazyFrame
        Plano lazy com a coluna ``year`` derivada da partição.

    Raises
    ------
    FileNotFoundError
        Se não houver nenhuma partição Parquet em ``silver_dir``.

    Examples
    --------
    >>> scan_silver(Path("data/silver/enem"), [2023])  # doctest: +SKIP
    """
    if not any(silver_dir.glob(f"{PARTITION_COLUMN}=*/*.parquet")):
        raise FileNotFoundError(
            f"Nenhuma partição silver em {silver_dir}. Execute a etapa 'ingest' primeiro."
        )
    pattern = (silver_dir / f"{PARTITION_COLUMN}=*" / "*.parquet").as_posix()
    lazy = (
        pl.scan_parquet(pattern, hive_partitioning=True)
        .with_columns(pl.col(PARTITION_COLUMN).cast(pl.Int16).alias(YEAR_COLUMN))
        .drop(PARTITION_COLUMN)
    )
    if years:
        lazy = lazy.filter(pl.col(YEAR_COLUMN).is_in(list(years)))
    return lazy


def read_parquet(path: Path) -> pl.DataFrame:
    """Lê um arquivo Parquet inteiro em memória.

    Parameters
    ----------
    path : Path
        Arquivo Parquet.

    Returns
    -------
    pl.DataFrame
        Conteúdo do arquivo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pl.read_parquet(path)
