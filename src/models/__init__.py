"""Modelos de regressão da nota do ENEM.

Módulos
-------
base
    Protocolo ``Regressor`` comum aos modelos.
linear_baseline
    ``LinearBaseline``: regressão linear (OLS, statsmodels) com one-hot e erros robustos.
gradient_boosting
    ``build_gradient_boosting``: XGBoost configurado por Pydantic.
persistence
    ``save_model`` e ``load_model`` (joblib).
"""
