"""Tema e paleta compartilhados por todas as figuras."""

import matplotlib.pyplot as plt
import seaborn as sns

# Paleta segura para daltonismo (Okabe-Ito).
PALETTE: list[str] = [
    "#0072B2",  # azul
    "#E69F00",  # laranja
    "#009E73",  # verde
    "#D55E00",  # vermelhão
    "#CC79A7",  # rosa
    "#56B4E9",  # azul-claro
    "#F0E442",  # amarelo
    "#000000",  # preto
]
HEATMAP_CMAP = "viridis"


def apply_theme() -> None:
    """Aplica o tema global (fundo claro, grade suave, paleta Okabe-Ito).

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", palette=PALETTE, context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
