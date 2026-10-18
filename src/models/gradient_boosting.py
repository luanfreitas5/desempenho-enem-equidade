"""Gradient boosting (XGBoost) configurado por ``GradientBoostingSettings``."""

from xgboost import XGBRegressor

from config.settings import GradientBoostingSettings


def build_gradient_boosting(params: GradientBoostingSettings, seed: int) -> XGBRegressor:
    """Cria um ``XGBRegressor`` não treinado com os hiperparâmetros validados.

    Parameters
    ----------
    params : GradientBoostingSettings
        Hiperparâmetros (``configs/model_params.yaml``).
    seed : int
        Semente (``random_state``) para reprodutibilidade.

    Returns
    -------
    XGBRegressor
        Modelo pronto para ``fit``. Valores ausentes (``NaN``) são tratados
        nativamente pelas árvores.

    Examples
    --------
    >>> model = build_gradient_boosting(GradientBoostingSettings(n_estimators=10), seed=0)
    """
    return XGBRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        min_child_weight=params.min_child_weight,
        n_jobs=params.n_jobs,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
    )
