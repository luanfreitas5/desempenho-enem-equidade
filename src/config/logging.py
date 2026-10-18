"""Logging com ``rich`` no console e arquivo diário com rotação."""

import logging
from datetime import date
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

FILE_FORMAT = "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    backup_count: int = 30,
) -> None:
    """Configura o logging do projeto (console com ``rich`` + arquivo diário).

    Parameters
    ----------
    level : str, optional
        Nível de log (``DEBUG``, ``INFO``, ``WARNING`` ...), by default "INFO".
    log_dir : Path or None, optional
        Pasta dos arquivos ``log_YYYY-MM-DD.log``. Se ``None``, apenas o
        console é configurado.
    backup_count : int, optional
        Quantidade de arquivos diários mantidos, by default 30.

    Raises
    ------
    ValueError
        Se ``level`` não for um nível de log válido.

    Examples
    --------
    >>> configure_logging("INFO", Path("logs"))
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Nível de log inválido: {level}")  # noqa: TRY004 (valor, não tipo)

    handlers: list[logging.Handler] = [RichHandler(rich_tracebacks=True, show_path=False)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / f"log_{date.today().isoformat()}.log"
        file_handler = TimedRotatingFileHandler(
            file_path, when="midnight", backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, format="%(message)s", handlers=handlers, force=True)
