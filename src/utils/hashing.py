"""Hash de arquivos para rastrear a versão dos dados."""

import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def hash_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Calcula o SHA-256 de um arquivo lendo-o em blocos (seguro para arquivos grandes).

    Parameters
    ----------
    path : Path
        Arquivo a ser verificado.
    chunk_size : int, optional
        Tamanho do bloco de leitura em bytes, by default 1 MiB.

    Returns
    -------
    str
        Hash SHA-256 em hexadecimal.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> len(hash_file(Path("pyproject.toml")))
    64
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
