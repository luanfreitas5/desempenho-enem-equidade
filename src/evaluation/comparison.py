"""Comparação estatística de modelos na validação cruzada."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TTestResult:
    """Resultado do teste t reamostrado corrigido."""

    statistic: float
    p_value: float
    degrees_of_freedom: int


def corrected_resampled_ttest(
    differences: Sequence[float], n_train: int, n_test: int
) -> TTestResult:
    """Teste t reamostrado corrigido (Nadeau & Bengio, 2003).

    Compara dois modelos avaliados nas mesmas dobras. A correção inflama a
    variância para considerar que os conjuntos de treino se sobrepõem, evitando
    o excesso de falsos positivos do teste t pareado ingênuo.

    Parameters
    ----------
    differences : Sequence[float]
        Diferença da métrica (modelo A - modelo B) em cada dobra.
    n_train : int
        Tamanho do conjunto de treino de cada dobra.
    n_test : int
        Tamanho do conjunto de teste de cada dobra.

    Returns
    -------
    TTestResult
        Estatística t, valor-p bilateral e graus de liberdade (dobras - 1).

    Raises
    ------
    ValueError
        Se houver menos de duas dobras.

    Examples
    --------
    >>> corrected_resampled_ttest([1.0, 1.1, 0.9, 1.0, 1.05], 800, 200).p_value < 0.05
    True
    """
    values = np.asarray(differences, dtype=float)
    n_folds = values.size
    if n_folds < 2:
        raise ValueError("São necessárias ao menos 2 dobras para o teste")
    mean = float(values.mean())
    variance = float(values.var(ddof=1))
    if variance == 0.0:
        statistic = 0.0 if mean == 0.0 else float(np.sign(mean) * np.inf)
    else:
        statistic = mean / float(np.sqrt((1 / n_folds + n_test / n_train) * variance))
    p_value = float(2 * stats.t.sf(abs(statistic), n_folds - 1))
    return TTestResult(statistic=statistic, p_value=p_value, degrees_of_freedom=n_folds - 1)
