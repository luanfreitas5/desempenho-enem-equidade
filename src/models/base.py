"""Protocolo comum aos regressores do projeto."""

from typing import Any, Protocol

import numpy as np


class Regressor(Protocol):
    """Interface mínima (estilo scikit-learn) exigida pela validação cruzada."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:  # noqa: N803 - convenção sklearn
        """Treina o modelo com a matriz ``X`` e o alvo ``y``."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803 - convenção sklearn
        """Prevê o alvo para a matriz ``X``."""
        ...
