"""Etapa 1 — ingestão: download → CSV lazy → Parquet silver particionado por ano."""

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from config.settings import Settings
from constants.artifacts import SILVER_MANIFEST_FILE
from constants.columns import PARTITION_COLUMN, RAW_YEAR_COLUMN
from data.downloader import MANIFEST_NAME, FetchFunction, download_file, download_microdata
from data.reader import scan_raw_csv
from data.writer import PART_FILE_NAME, sink_year_partition
from exceptions.data import DataValidationError
from preprocessing.cleaning import transform_raw_to_silver
from schemas.dataset import validate_silver
from utils.hashing import hash_file
from utils.serialization import read_json, write_json

logger = logging.getLogger(__name__)

SILVER_VALIDATION_ROWS = 100_000


def validate_silver_partition(partition_file: Path, year: int) -> int:
    """Valida a partição silver gravada e devolve o número de linhas.

    Parameters
    ----------
    partition_file : Path
        Arquivo Parquet da partição.
    year : int
        Ano esperado da partição.

    Returns
    -------
    int
        Número de linhas da partição.

    Raises
    ------
    DataValidationError
        Se a partição contiver anos diferentes do esperado ou violar o
        contrato de dados (validado nas primeiras ``SILVER_VALIDATION_ROWS`` linhas).
    """
    lazy = pl.scan_parquet(partition_file)
    years_found = lazy.select(pl.col(RAW_YEAR_COLUMN).drop_nulls().unique()).collect().to_series()
    if years_found.to_list() != [year]:
        raise DataValidationError(
            f"Partição ano={year} contém anos {years_found.to_list()} em {RAW_YEAR_COLUMN}"
        )
    validate_silver(lazy.head(SILVER_VALIDATION_ROWS).collect())
    return int(lazy.select(pl.len()).collect().item())


def run_ingestion(
    settings: Settings,
    years: Sequence[int] | None = None,
    force: bool = False,
    fetch: FetchFunction = download_file,
) -> list[Path]:
    """Executa a ingestão dos microdados do INEP até a camada silver.

    Parameters
    ----------
    settings : Settings
        Configuração validada do projeto.
    years : Sequence[int] or None, optional
        Anos a processar; ``None`` usa ``settings.ingestion.years``.
    force : bool, optional
        Se ``True``, baixa e reprocessa mesmo com cache, by default False.
    fetch : FetchFunction, optional
        Função de download (injetável para testes).

    Returns
    -------
    list[Path]
        Arquivos Parquet das partições silver.

    Raises
    ------
    DataDownloadError
        Se o download ou a extração falharem.
    DataValidationError
        Se a partição gerada violar o contrato de dados.

    Notes
    -----
    A etapa é idempotente: se a partição silver do ano já existe, nada é
    baixado nem reprocessado (use ``force=True`` para refazer). O bronze
    (CSV bruto) nunca é alterado.
    """
    ingestion = settings.ingestion
    raw_dir = settings.paths.resolved("raw_dir")
    silver_dir = settings.paths.resolved("silver_dir")
    outputs: list[Path] = []
    for year in years or ingestion.years:
        target = silver_dir / f"{PARTITION_COLUMN}={year}" / PART_FILE_NAME
        if target.is_file() and not force:
            logger.info("Partição silver de %d já existe: %s", year, target)
            outputs.append(target)
            continue
        csv_path = download_microdata(
            year,
            raw_dir,
            ingestion.base_url,
            timeout=ingestion.timeout_seconds,
            keep_zip=ingestion.keep_zip,
            force=force,
            fetch=fetch,
        )
        raw = scan_raw_csv(csv_path, ingestion.separator, ingestion.encoding)
        written = sink_year_partition(transform_raw_to_silver(raw), silver_dir, year)
        n_rows = validate_silver_partition(written, year)
        raw_manifest = csv_path.parent / MANIFEST_NAME
        write_json(
            {
                "year": year,
                "rows": n_rows,
                "parquet_sha256": hash_file(written),
                "source_csv_sha256": read_json(raw_manifest).get("csv_sha256")
                if raw_manifest.is_file()
                else None,
            },
            written.parent / SILVER_MANIFEST_FILE,
        )
        logger.info("Silver %d: %d linhas validadas", year, n_rows)
        outputs.append(written)
    return outputs
