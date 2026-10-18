"""Reprodutibilidade: fixa todas as fontes de aleatoriedade."""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Fixa as sementes de ``random``, NumPy e ``PYTHONHASHSEED``.

    Parameters
    ----------
    seed : int, optional
        Semente global, by default 42.

    Notes
    -----
    ``PYTHONHASHSEED`` definido dentro do processo só afeta subprocessos; para
    afetar o próprio interpretador ele precisa ser exportado antes da execução
    (o ``Makefile`` faz isso).

    Examples
    --------
    >>> seed_everything(42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - semente global para bibliotecas legadas
    logger.info("Sementes fixadas em %d", seed)
