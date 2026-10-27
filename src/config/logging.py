"""Configuração única de logging: console ``rich`` e arquivo diário com rotação."""

import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

from config.paths import CONFIG_DIR, ROOT
from exceptions.configuration import ConfigurationError

CONSOLE = Console(stderr=True)  # compartilhado por RichHandler e rich.progress


def create_console_handler() -> RichHandler:
    """Cria o handler de console ligado ao console compartilhado.

    Returns
    -------
    RichHandler
        Handler que escreve no mesmo ``Console`` das barras de progresso.
    """
    return RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=True, markup=False)


def _rename_rotated(default_name: str) -> str:
    """Converte ``logs/log.log.2026-09-27`` em ``logs/log_2026-09-27.log``."""
    path = Path(default_name)
    stem, date_str = path.name.split(".log.")
    return str(path.with_name(f"{stem}_{date_str}.log"))


def create_daily_file_handler(filename: str, backup_count: int = 30) -> TimedRotatingFileHandler:
    """Cria o handler de arquivo com rotação diária à meia-noite.

    Parameters
    ----------
    filename : str
        Caminho do arquivo ativo, relativo à raiz do projeto.
    backup_count : int, optional
        Quantidade de arquivos diários mantidos, by default 30.

    Returns
    -------
    TimedRotatingFileHandler
        Handler que rotaciona para ``log_YYYY-MM-DD.log``.
    """
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # dictConfig falha se o diretório não existe
    handler = TimedRotatingFileHandler(
        path, when="midnight", backupCount=backup_count, encoding="utf-8"
    )
    handler.namer = _rename_rotated
    return handler


def configure_logging(level: str | None = None, config_path: Path | None = None) -> None:
    """Aplica a configuração de logging do YAML. Chamar apenas no ponto de entrada.

    Parameters
    ----------
    level : str or None, optional
        Sobrescreve o nível do root (ex.: ``--log-level DEBUG``), by default None.
    config_path : Path or None, optional
        YAML no formato dictConfig, by default ``configs/logging.yaml``.

    Raises
    ------
    ConfigurationError
        Se o YAML não existir ou o nível de log for inválido.

    Examples
    --------
    >>> configure_logging(level="DEBUG")  # doctest: +SKIP
    """
    path = config_path or CONFIG_DIR / "logging.yaml"
    if not path.is_file():
        raise ConfigurationError(f"Arquivo de configuração não encontrado: {path}")
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    if level is not None:
        level = level.upper()
        if not isinstance(logging.getLevelName(level), int):
            raise ConfigurationError(f"Nível de log inválido: {level}")
        config["root"]["level"] = level
    logging.config.dictConfig(config)  # substitui os handlers do root: chamar de novo não duplica
    logging.captureWarnings(True)  # warnings.warn() passa pelos mesmos handlers
