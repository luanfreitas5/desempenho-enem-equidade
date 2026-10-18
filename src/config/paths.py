"""Caminhos centralizados do projeto, sempre com ``pathlib.Path``."""

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = ROOT / "configs"


def resolve_path(path: Path | str) -> Path:
    """Resolve um caminho relativo à raiz do projeto.

    Parameters
    ----------
    path : Path or str
        Caminho absoluto ou relativo à raiz do projeto.

    Returns
    -------
    Path
        Caminho absoluto. Caminhos já absolutos são devolvidos sem alteração.

    Examples
    --------
    >>> resolve_path("data/bronze").is_absolute()
    True
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate
