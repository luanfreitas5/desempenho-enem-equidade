"""Métricas de regressão e intervalos de confiança (t de Student e bootstrap)."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

MetricFunction = Callable[[np.ndarray, np.ndarray], float]


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raiz do erro quadrático médio (na escala da nota, de 0 a 1000).

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.

    Returns
    -------
    float
        RMSE.

    Examples
    --------
    >>> calculate_rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
    1.4142135623730951
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro absoluto médio.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.

    Returns
    -------
    float
        MAE.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinação R².

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.

    Returns
    -------
    float
        R²; ``NaN`` se os valores observados forem constantes.
    """
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


def calculate_mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro médio com sinal (previsto - observado): positivo = superestima a nota.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.

    Returns
    -------
    float
        Viés médio do modelo.
    """
    return float(np.mean(y_pred - y_true))


def calculate_mean_target(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # noqa: ARG001
    """Média da nota observada (ignora ``y_pred``; assinatura compatível com métricas).

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos (não utilizado).

    Returns
    -------
    float
        Nota média observada.
    """
    return float(np.mean(y_true))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula RMSE, MAE e R² de uma vez.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.

    Returns
    -------
    dict[str, float]
        Chaves ``rmse``, ``mae`` e ``r2``.
    """
    return {
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
    }


@dataclass(frozen=True)
class ScoreSummary:
    """Resumo de uma métrica ao longo das dobras: média, desvio e IC."""

    mean: float
    std: float
    ci_low: float
    ci_high: float


def summarize_scores(values: Sequence[float], confidence_level: float = 0.95) -> ScoreSummary:
    """Resume métricas por dobra com média e IC pela distribuição t de Student.

    Parameters
    ----------
    values : Sequence[float]
        Valor da métrica em cada dobra (ao menos duas).
    confidence_level : float, optional
        Nível de confiança do intervalo, by default 0.95.

    Returns
    -------
    ScoreSummary
        Média, desvio-padrão amostral e limites do IC.

    Raises
    ------
    ValueError
        Se houver menos de dois valores.

    Notes
    -----
    As dobras compartilham dados de treino, então o IC tende a ser otimista;
    use o teste corrigido de ``evaluation.comparison`` para comparar modelos.

    Examples
    --------
    >>> round(summarize_scores([1.0, 2.0, 3.0]).mean, 1)
    2.0
    """
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("São necessários ao menos 2 valores para calcular o IC")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    margin = float(stats.t.ppf((1 + confidence_level) / 2, array.size - 1)) * std
    margin /= float(np.sqrt(array.size))
    return ScoreSummary(mean=mean, std=std, ci_low=mean - margin, ci_high=mean + margin)


def compute_percentile_interval(
    samples: np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Intervalo percentil de uma amostra bootstrap.

    Parameters
    ----------
    samples : np.ndarray
        Réplicas bootstrap da estatística.
    confidence_level : float, optional
        Nível de confiança, by default 0.95.

    Returns
    -------
    tuple[float, float]
        Limites inferior e superior.

    Examples
    --------
    >>> compute_percentile_interval(np.arange(101.0), 0.9)
    (5.0, 95.0)
    """
    tail = (1 - confidence_level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    statistics: Mapping[str, MetricFunction],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reamostra os pares (observado, previsto) com reposição e recalcula as estatísticas.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    y_pred : np.ndarray
        Valores previstos.
    statistics : Mapping[str, MetricFunction]
        Estatísticas a recalcular em cada réplica, por nome.
    n_bootstrap : int
        Número de réplicas.
    seed : int
        Semente do gerador de números aleatórios.

    Returns
    -------
    dict[str, np.ndarray]
        Para cada estatística, um vetor com ``n_bootstrap`` réplicas.

    Examples
    --------
    >>> y = np.arange(10.0)
    >>> out = bootstrap_statistics(y, y, {"rmse": calculate_rmse}, 20, seed=0)
    >>> float(out["rmse"].max())
    0.0
    """
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    replicates = {name: np.empty(n_bootstrap) for name in statistics}
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_rows, size=n_rows)
        resampled_true, resampled_pred = y_true[sample], y_pred[sample]
        for name, function in statistics.items():
            replicates[name][index] = function(resampled_true, resampled_pred)
    return replicates


def bootstrap_metric_difference(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: MetricFunction,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Réplicas bootstrap pareadas de ``metric(A) - metric(B)`` nas mesmas linhas.

    Parameters
    ----------
    y_true : np.ndarray
        Valores observados.
    prediction_a : np.ndarray
        Previsões do modelo A.
    prediction_b : np.ndarray
        Previsões do modelo B.
    metric : MetricFunction
        Métrica a comparar (por exemplo, ``calculate_rmse``).
    n_bootstrap : int
        Número de réplicas.
    seed : int
        Semente do gerador de números aleatórios.

    Returns
    -------
    np.ndarray
        Vetor com ``n_bootstrap`` diferenças. Se o IC não contiver zero, a
        diferença entre os modelos é estatisticamente distinguível.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    differences = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_rows, size=n_rows)
        observed = y_true[sample]
        differences[index] = metric(observed, prediction_a[sample]) - metric(
            observed, prediction_b[sample]
        )
    return differences
