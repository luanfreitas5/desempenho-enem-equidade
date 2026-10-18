"""Leitura e escrita de resultados em JSON e CSV."""

import json
from pathlib import Path
from typing import Any

import polars as pl


def write_json(data: dict[str, Any], path: Path) -> Path:
    """Grava um dicionário como JSON UTF-8 (cria a pasta se necessário).

    Parameters
    ----------
    data : dict[str, Any]
        Conteúdo serializável em JSON.
    path : Path
        Arquivo de destino.

    Returns
    -------
    Path
        O próprio ``path`` gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    """Lê um arquivo JSON UTF-8.

    Parameters
    ----------
    path : Path
        Arquivo JSON.

    Returns
    -------
    dict[str, Any]
        Conteúdo do arquivo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(table: pl.DataFrame, path: Path) -> Path:
    """Grava uma tabela de resultados em CSV (cria a pasta se necessário).

    Parameters
    ----------
    table : pl.DataFrame
        Tabela a gravar.
    path : Path
        Arquivo CSV de destino.

    Returns
    -------
    Path
        O próprio ``path`` gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path)
    return path
