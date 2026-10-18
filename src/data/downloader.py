"""Download idempotente dos microdados do ENEM (INEP) com cache local.

Os arquivos brutos vão para ``data/bronze/raw/<ano>/`` e nunca são modificados.
Um ``manifest.json`` por ano registra URL, tamanho e SHA-256 do CSV; se o CSV e
o manifest coincidem, o download é pulado (execução idempotente).
"""

import logging
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from exceptions.data import DataDownloadError
from utils.hashing import hash_file
from utils.progress import create_progress
from utils.serialization import read_json, write_json

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
FetchFunction = Callable[[str, Path, int], None]


def build_download_url(base_url: str, year: int) -> str:
    """Monta a URL de download do ZIP de microdados de um ano.

    Parameters
    ----------
    base_url : str
        Modelo de URL com o marcador ``{year}``.
    year : int
        Ano da edição do ENEM.

    Returns
    -------
    str
        URL completa.

    Examples
    --------
    >>> build_download_url("https://exemplo.gov.br/enem_{year}.zip", 2023)
    'https://exemplo.gov.br/enem_2023.zip'
    """
    return base_url.format(year=year)


def build_csv_name(year: int) -> str:
    """Devolve o nome padrão do CSV de microdados de um ano.

    Parameters
    ----------
    year : int
        Ano da edição do ENEM.

    Returns
    -------
    str
        Nome do arquivo, por exemplo ``MICRODADOS_ENEM_2023.csv``.
    """
    return f"MICRODADOS_ENEM_{year}.csv"


def download_file(url: str, destination: Path, timeout: int) -> None:  # pragma: no cover
    """Baixa uma URL HTTPS para ``destination`` (via arquivo temporário ``.part``).

    Parameters
    ----------
    url : str
        URL HTTPS de origem.
    destination : Path
        Arquivo de destino.
    timeout : int
        Tempo limite de rede em segundos.

    Raises
    ------
    DataDownloadError
        Se a URL não for HTTPS ou ocorrer falha de rede.
    """
    if not url.startswith("https://"):
        raise DataDownloadError(f"Somente URLs HTTPS são permitidas: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "desempenho-enem-equidade"})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,  # nosec B310
            partial.open("wb") as handle,
            create_progress() as progress,
        ):
            total = int(response.headers.get("Content-Length") or 0) or None
            task = progress.add_task(f"Baixando {destination.name}", total=total)
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                progress.advance(task, len(chunk))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise DataDownloadError(f"Falha ao baixar {url}: {error}") from error
    partial.replace(destination)


def extract_microdata_csv(zip_path: Path, year: int, target_dir: Path) -> Path:
    """Extrai do ZIP o CSV de microdados do ano, sem respeitar subpastas do arquivo.

    Parameters
    ----------
    zip_path : Path
        ZIP baixado do INEP.
    year : int
        Ano da edição do ENEM.
    target_dir : Path
        Pasta onde o CSV será gravado.

    Returns
    -------
    Path
        Caminho do CSV extraído.

    Raises
    ------
    DataDownloadError
        Se o ZIP for inválido ou não contiver o CSV esperado.
    """
    expected = build_csv_name(year).lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            member = next(
                (name for name in archive.namelist() if Path(name).name.lower() == expected),
                None,
            )
            if member is None:
                raise DataDownloadError(f"{build_csv_name(year)} não encontrado em {zip_path}")
            csv_path = target_dir / build_csv_name(year)
            # Usa apenas o nome base do membro: evita escrita fora de target_dir (zip-slip).
            with archive.open(member) as source, csv_path.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    except zipfile.BadZipFile as error:
        raise DataDownloadError(f"ZIP inválido: {zip_path}") from error
    return csv_path


def is_cached(csv_path: Path, manifest_path: Path) -> bool:
    """Indica se o CSV já está em cache e consistente com o manifest.

    Parameters
    ----------
    csv_path : Path
        CSV esperado.
    manifest_path : Path
        Manifest do ano.

    Returns
    -------
    bool
        ``True`` se CSV e manifest existem e o tamanho registrado confere.
    """
    if not (csv_path.is_file() and manifest_path.is_file()):
        return False
    manifest = read_json(manifest_path)
    return manifest.get("csv_size_bytes") == csv_path.stat().st_size


def download_microdata(
    year: int,
    raw_dir: Path,
    base_url: str,
    *,
    timeout: int = 120,
    keep_zip: bool = False,
    force: bool = False,
    fetch: FetchFunction = download_file,
) -> Path:
    """Baixa e extrai os microdados de um ano (idempotente, com cache local).

    Parameters
    ----------
    year : int
        Ano da edição do ENEM.
    raw_dir : Path
        Pasta bronze/raw; cada ano usa uma subpasta.
    base_url : str
        Modelo de URL com o marcador ``{year}``.
    timeout : int, optional
        Tempo limite de rede em segundos, by default 120.
    keep_zip : bool, optional
        Se ``True``, mantém o ZIP após a extração, by default False.
    force : bool, optional
        Se ``True``, ignora o cache e baixa novamente, by default False.
    fetch : FetchFunction, optional
        Função de download (injetável para testes), by default ``download_file``.

    Returns
    -------
    Path
        Caminho do CSV bruto.

    Raises
    ------
    DataDownloadError
        Se o download ou a extração falharem.

    Examples
    --------
    >>> download_microdata(2023, Path("data/bronze/raw"), URL)  # doctest: +SKIP
    PosixPath('data/bronze/raw/2023/MICRODADOS_ENEM_2023.csv')
    """
    year_dir = raw_dir / str(year)
    csv_path = year_dir / build_csv_name(year)
    manifest_path = year_dir / MANIFEST_NAME

    if not force and is_cached(csv_path, manifest_path):
        logger.info("Microdados de %d já em cache: %s", year, csv_path)
        return csv_path

    url = build_download_url(base_url, year)
    zip_path = year_dir / f"microdados_enem_{year}.zip"
    logger.info("Baixando microdados de %d de %s", year, url)
    fetch(url, zip_path, timeout)
    csv_path = extract_microdata_csv(zip_path, year, year_dir)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    write_json(
        {
            "year": year,
            "url": url,
            "csv_file": csv_path.name,
            "csv_size_bytes": csv_path.stat().st_size,
            "csv_sha256": hash_file(csv_path),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        },
        manifest_path,
    )
    logger.info("Microdados de %d gravados em %s", year, csv_path)
    return csv_path
