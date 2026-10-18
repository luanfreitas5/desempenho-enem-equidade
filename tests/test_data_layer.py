"""Testes de download, leitura, gravação, amostragem e divisão de dados."""

from pathlib import Path

import polars as pl
import pytest

from constants.columns import INGEST_COLUMNS
from data.downloader import (
    MANIFEST_NAME,
    build_csv_name,
    build_download_url,
    download_microdata,
    extract_microdata_csv,
)
from data.loader import read_parquet, scan_silver
from data.reader import scan_raw_csv
from data.sampler import sample_rows
from data.splitter import split_train_test
from data.writer import sink_year_partition, write_parquet
from exceptions.data import DataDownloadError, InsufficientDataError
from factories import build_raw_frame, write_microdata_zip, write_raw_csv
from preprocessing.cleaning import transform_raw_to_silver
from schemas.dataset import validate_silver

BASE_URL = "https://exemplo.gov.br/microdados_enem_{year}.zip"


def test_build_download_url_and_csv_name():
    """URL e nome do CSV seguem o padrão do INEP."""
    assert build_download_url(BASE_URL, 2023) == "https://exemplo.gov.br/microdados_enem_2023.zip"
    assert build_csv_name(2023) == "MICRODADOS_ENEM_2023.csv"


def test_extract_microdata_csv_ignores_subfolders(tmp_path: Path):
    """O CSV é extraído para a pasta de destino, sem recriar subpastas do ZIP."""
    zip_path = write_microdata_zip(tmp_path / "a.zip", build_raw_frame(10), 2023)
    csv_path = extract_microdata_csv(zip_path, 2023, tmp_path / "out")
    assert csv_path == tmp_path / "out" / "MICRODADOS_ENEM_2023.csv"
    assert csv_path.is_file()


def test_extract_microdata_csv_errors(tmp_path: Path):
    """ZIP sem o CSV esperado ou inválido gera DataDownloadError."""
    zip_path = write_microdata_zip(tmp_path / "a.zip", build_raw_frame(10), 2023)
    with pytest.raises(DataDownloadError):
        extract_microdata_csv(zip_path, 2020, tmp_path / "out")
    bad = tmp_path / "ruim.zip"
    bad.write_bytes(b"isto nao e um zip")
    with pytest.raises(DataDownloadError):
        extract_microdata_csv(bad, 2023, tmp_path / "out")


@pytest.mark.smoke
def test_download_microdata_is_idempotent(tmp_path: Path):
    """A segunda chamada usa o cache e só ``force`` baixa novamente."""
    calls: list[str] = []

    def fake_fetch(url: str, destination: Path, timeout: int) -> None:
        calls.append(url)
        write_microdata_zip(destination, build_raw_frame(20), 2023)

    raw_dir = tmp_path / "raw"
    first = download_microdata(2023, raw_dir, BASE_URL, fetch=fake_fetch)
    second = download_microdata(2023, raw_dir, BASE_URL, fetch=fake_fetch)
    assert first == second
    assert len(calls) == 1
    assert (raw_dir / "2023" / MANIFEST_NAME).is_file()
    assert not list((raw_dir / "2023").glob("*.zip"))  # ZIP removido após extração

    download_microdata(2023, raw_dir, BASE_URL, fetch=fake_fetch, force=True)
    assert len(calls) == 2


def test_scan_raw_csv_missing_file(tmp_path: Path):
    """Arquivo inexistente gera FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        scan_raw_csv(tmp_path / "nao_existe.csv")


@pytest.mark.smoke
def test_latin1_csv_to_partitioned_parquet(tmp_path: Path):
    """CSV latin-1 → silver Parquet particionado por ano, válido no contrato."""
    csv_path = write_raw_csv(build_raw_frame(200, year=2023), tmp_path / "raw.csv")
    lazy = scan_raw_csv(csv_path)
    assert isinstance(lazy, pl.LazyFrame)

    silver_dir = tmp_path / "silver"
    written = sink_year_partition(transform_raw_to_silver(lazy), silver_dir, 2023)
    assert written == silver_dir / "ano=2023" / "part-0.parquet"
    assert not list(silver_dir.rglob("*.tmp"))

    loaded = scan_silver(silver_dir, [2023]).collect()
    assert loaded.height == 200
    assert set(INGEST_COLUMNS) <= set(loaded.columns)
    assert "NU_INSCRICAO" not in loaded.columns  # minimização de dados
    assert loaded["year"].dtype == pl.Int16
    validate_silver(pl.read_parquet(written))


def test_scan_silver_without_partitions(tmp_path: Path):
    """Sem partições, a leitura orienta o usuário a rodar a ingestão."""
    with pytest.raises(FileNotFoundError, match="ingest"):
        scan_silver(tmp_path)


def test_scan_silver_filters_years(tmp_path: Path):
    """O filtro de anos devolve apenas as partições pedidas."""
    for year in (2022, 2023):
        frame = build_raw_frame(30, year=year, seed=year)
        csv_path = write_raw_csv(frame, tmp_path / f"{year}.csv")
        sink_year_partition(transform_raw_to_silver(scan_raw_csv(csv_path)), tmp_path / "s", year)
    only_2022 = scan_silver(tmp_path / "s", [2022]).collect()
    assert only_2022["year"].unique().to_list() == [2022]
    assert scan_silver(tmp_path / "s").collect().height == 60


def test_write_and_read_parquet(tmp_path: Path):
    """Gravação e leitura de Parquet preservam os dados."""
    frame = pl.DataFrame({"a": [1, 2, 3]})
    path = write_parquet(frame, tmp_path / "sub" / "x.parquet")
    assert read_parquet(path).equals(frame)
    with pytest.raises(FileNotFoundError):
        read_parquet(tmp_path / "ausente.parquet")


def test_sample_rows_reproducible_and_bounded():
    """A amostra é reprodutível; tamanho maior que a tabela devolve tudo."""
    frame = pl.DataFrame({"a": range(100)})
    assert sample_rows(frame, 10, seed=1).equals(sample_rows(frame, 10, seed=1))
    assert sample_rows(frame, 10, seed=1).height == 10
    assert sample_rows(frame, None, seed=1).height == 100
    assert sample_rows(frame, 500, seed=1).height == 100


def test_split_train_test_disjoint_and_errors():
    """Treino e teste são disjuntos, cobrem tudo e respeitam test_size."""
    frame = pl.DataFrame({"a": range(50)})
    train, test = split_train_test(frame, 0.2, seed=3)
    assert (train.height, test.height) == (40, 10)
    assert set(train["a"]).isdisjoint(set(test["a"]))
    assert sorted([*train["a"], *test["a"]]) == list(range(50))
    with pytest.raises(InsufficientDataError):
        split_train_test(pl.DataFrame({"a": [1]}), 0.2, seed=0)
