"""Persistência de modelos treinados (joblib)."""

import logging
from pathlib import Path
from typing import Any

import joblib

from utils.hashing import hash_file

logger = logging.getLogger(__name__)


def save_model(model: Any, path: Path) -> str:
    """Salva um modelo com joblib e devolve o hash SHA-256 do arquivo.

    Parameters
    ----------
    model : Any
        Modelo treinado.
    path : Path
        Arquivo de destino (a pasta é criada se necessário).

    Returns
    -------
    str
        SHA-256 do arquivo salvo, útil para rastrear a versão do modelo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    digest = hash_file(path)
    logger.info("Modelo salvo em %s (sha256=%s)", path, digest[:12])
    return digest


def load_model(path: Path) -> Any:
    """Carrega um modelo salvo com joblib.

    Parameters
    ----------
    path : Path
        Arquivo gerado por ``save_model``. Carregue apenas arquivos confiáveis:
        o formato é baseado em pickle.

    Returns
    -------
    Any
        Modelo desserializado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    return joblib.load(path)
