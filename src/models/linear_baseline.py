"""Regressão linear de referência (OLS, statsmodels) sobre o questionário em one-hot."""

from collections.abc import Sequence

import numpy as np
import polars as pl
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder
from statsmodels.regression.linear_model import RegressionResultsWrapper

from exceptions.model import ModelNotFittedError


class LinearBaseline:
    """Regressão linear de referência com variáveis categóricas em one-hot.

    Cada resposta do questionário vira um conjunto de indicadoras (a primeira
    categoria é a referência). O ajuste é OLS; os intervalos de confiança dos
    coeficientes usam erros-padrão robustos à heterocedasticidade (HC1).

    Parameters
    ----------
    feature_names : Sequence[str]
        Nomes das colunas de ``X`` (na mesma ordem), usados nos coeficientes.

    Examples
    --------
    >>> model = LinearBaseline(["Q001"])
    >>> X = np.array([["A"], ["B"], ["A"], ["B"]], dtype=object)
    >>> model.fit(X, np.array([1.0, 2.0, 1.0, 2.0])).predict(X).round(1).tolist()
    [1.0, 2.0, 1.0, 2.0]
    """

    def __init__(self, feature_names: Sequence[str]) -> None:
        """Inicializa o modelo sem ajustá-lo."""
        self.feature_names = list(feature_names)
        self._encoder: OneHotEncoder | None = None
        self._result: RegressionResultsWrapper | None = None

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Aplica o one-hot e adiciona o intercepto."""
        if self._encoder is None:
            raise ModelNotFittedError("LinearBaseline usado antes de fit()")
        return np.asarray(sm.add_constant(self._encoder.transform(X), has_constant="add"))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBaseline":  # noqa: N803
        """Ajusta o OLS.

        Parameters
        ----------
        X : np.ndarray
            Matriz de texto ``(n, p)`` com as respostas do questionário.
        y : np.ndarray
            Nota média (alvo).

        Returns
        -------
        LinearBaseline
            O próprio modelo, ajustado.
        """
        self._encoder = OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False, dtype=np.float64
        )
        self._encoder.fit(X)
        self._result = sm.OLS(y, self._design_matrix(X)).fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Prevê a nota média.

        Parameters
        ----------
        X : np.ndarray
            Matriz de texto ``(n, p)``. Categorias não vistas no treino são ignoradas.

        Returns
        -------
        np.ndarray
            Previsões de formato ``(n,)``.

        Raises
        ------
        ModelNotFittedError
            Se o modelo não tiver sido ajustado.
        """
        if self._result is None:
            raise ModelNotFittedError("LinearBaseline usado antes de fit()")
        return np.asarray(self._result.predict(self._design_matrix(X)))

    def summarize_coefficients(self, alpha: float = 0.05) -> pl.DataFrame:
        """Tabela de coeficientes com erros-padrão robustos (HC1) e IC.

        Parameters
        ----------
        alpha : float, optional
            Nível de significância do intervalo, by default 0.05 (IC de 95%).

        Returns
        -------
        pl.DataFrame
            Colunas ``term``, ``coefficient``, ``std_error``, ``ci_low``,
            ``ci_high`` e ``p_value``.

        Raises
        ------
        ModelNotFittedError
            Se o modelo não tiver sido ajustado.
        """
        if self._result is None or self._encoder is None:
            raise ModelNotFittedError("LinearBaseline usado antes de fit()")
        robust = self._result.get_robustcov_results(cov_type="HC1")
        interval = np.asarray(robust.conf_int(alpha=alpha))
        encoded_names = self._encoder.get_feature_names_out(self.feature_names)
        terms = ["const", *(str(name) for name in encoded_names)]
        return pl.DataFrame(
            {
                "term": terms,
                "coefficient": np.asarray(robust.params),
                "std_error": np.asarray(robust.bse),
                "ci_low": interval[:, 0],
                "ci_high": interval[:, 1],
                "p_value": np.asarray(robust.pvalues),
            }
        )
