"""Testes de modelos, métricas, validação cruzada, comparação, equidade e SHAP."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.linear_model import LinearRegression

from config.settings import GradientBoostingSettings
from evaluation.comparison import corrected_resampled_ttest
from evaluation.metrics import (
    bootstrap_metric_difference,
    bootstrap_statistics,
    calculate_mae,
    calculate_mean_error,
    calculate_mean_target,
    calculate_r2,
    calculate_rmse,
    compute_percentile_interval,
    compute_regression_metrics,
    summarize_scores,
)
from exceptions.model import ModelNotFittedError
from explainability.shap_analysis import (
    compute_shap_values,
    summarize_global_importance,
    summarize_shap_by_group,
)
from fairness.group_metrics import analyze_group_equity
from models.gradient_boosting import build_gradient_boosting
from models.linear_baseline import LinearBaseline
from models.persistence import load_model, save_model
from training.cross_validation import cross_validate_model, results_to_frame

RNG_SEED = 42


def make_categorical_problem(n_rows: int = 600):
    """Problema sintético: nota = efeito aditivo de duas questões categóricas + ruído leve."""
    rng = np.random.default_rng(RNG_SEED)
    first = rng.choice(["A", "B", "C"], size=n_rows)
    second = rng.choice(["X", "Y", "Z"], size=n_rows)
    effects_first = {"A": 0.0, "B": 50.0, "C": 120.0}
    effects_second = {"X": 0.0, "Y": 30.0, "Z": -20.0}
    y = (
        400.0
        + np.array([effects_first[v] for v in first])
        + np.array([effects_second[v] for v in second])
        + rng.normal(0, 1.0, n_rows)
    )
    return np.column_stack([first, second]).astype(object), y


# --- Modelo linear ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_linear_baseline_recovers_additive_effects():
    """O OLS com one-hot recupera um efeito aditivo conhecido (R² ≈ 1)."""
    X, y = make_categorical_problem()
    model = LinearBaseline(["Q001", "Q002"]).fit(X, y)
    assert calculate_r2(y, model.predict(X)) > 0.99
    table = model.summarize_coefficients()
    assert table.columns == ["term", "coefficient", "std_error", "ci_low", "ci_high", "p_value"]
    assert table.height == 1 + 2 + 2  # intercepto + (3-1) + (3-1) indicadoras
    effect_b = table.filter(pl.col("term") == "Q001_B")["coefficient"][0]
    assert effect_b == pytest.approx(50.0, abs=1.0)


def test_linear_baseline_tolerates_unseen_category_and_requires_fit():
    """Categorias novas são ignoradas na previsão; usar sem ajustar falha."""
    X, y = make_categorical_problem()
    model = LinearBaseline(["Q001", "Q002"])
    with pytest.raises(ModelNotFittedError):
        model.predict(X)
    with pytest.raises(ModelNotFittedError):
        model.summarize_coefficients()
    model.fit(X, y)
    unseen = np.array([["Z", "X"]], dtype=object)
    assert model.predict(unseen).shape == (1,)


def test_gradient_boosting_fits_and_persists(tmp_path: Path):
    """O XGBoost aprende uma relação monotônica e o modelo salvo prevê igual."""
    rng = np.random.default_rng(RNG_SEED)
    X = rng.integers(0, 10, size=(400, 3)).astype(np.float32)
    y = 20.0 * X[:, 0] + rng.normal(0, 1.0, 400)
    model = build_gradient_boosting(
        GradientBoostingSettings(n_estimators=40, max_depth=3, n_jobs=1, min_child_weight=1.0),
        seed=RNG_SEED,
    )
    model.fit(X, y)
    assert calculate_r2(y, model.predict(X)) > 0.9
    digest = save_model(model, tmp_path / "m" / "gb.joblib")
    assert len(digest) == 64
    assert np.allclose(load_model(tmp_path / "m" / "gb.joblib").predict(X), model.predict(X))
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "ausente.joblib")


@pytest.mark.ml
def test_gradient_boosting_directional_expectation():
    """Comportamental: mais renda (código maior) não reduz a nota prevista em média."""
    rng = np.random.default_rng(RNG_SEED)
    X = rng.integers(0, 17, size=(500, 4)).astype(np.float32)
    y = 10.0 * X[:, 0] + rng.normal(0, 5.0, 500)
    model = build_gradient_boosting(
        GradientBoostingSettings(n_estimators=60, max_depth=3, n_jobs=1, min_child_weight=1.0),
        seed=RNG_SEED,
    ).fit(X, y)
    low, high = X.copy(), X.copy()
    low[:, 0], high[:, 0] = 1.0, 15.0
    assert model.predict(high).mean() > model.predict(low).mean()


# --- Métricas ---------------------------------------------------------------------------------
def test_metrics_known_values():
    """RMSE, MAE, R² e viés em um caso calculado à mão."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 6.0])
    assert calculate_rmse(y_true, y_pred) == pytest.approx(1.0)
    assert calculate_mae(y_true, y_pred) == pytest.approx(0.5)
    assert calculate_mean_error(y_true, y_pred) == pytest.approx(0.5)
    assert calculate_mean_target(y_true, y_pred) == pytest.approx(2.5)
    assert calculate_r2(y_true, y_true) == 1.0
    assert np.isnan(calculate_r2(np.ones(3), np.zeros(3)))
    assert compute_regression_metrics(y_true, y_pred)["mae"] == pytest.approx(0.5)


finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)


@given(st.lists(st.tuples(finite, finite), min_size=1, max_size=40))
def test_rmse_is_at_least_mae(pairs):
    """Invariante: RMSE ≥ MAE ≥ 0 para quaisquer valores."""
    y_true = np.array([a for a, _ in pairs])
    y_pred = np.array([b for _, b in pairs])
    assert calculate_rmse(y_true, y_pred) + 1e-9 >= calculate_mae(y_true, y_pred) >= 0.0


def test_summarize_scores_and_errors():
    """A média está dentro do IC; menos de 2 valores é erro."""
    summary = summarize_scores([10.0, 12.0, 11.0, 13.0, 9.0])
    assert summary.ci_low < summary.mean < summary.ci_high
    assert summary.mean == pytest.approx(11.0)
    with pytest.raises(ValueError, match="ao menos 2"):
        summarize_scores([1.0])


def test_percentile_interval_and_bootstrap_reproducibility():
    """Intervalo percentil conhecido; bootstrap reprodutível e cobrindo a média."""
    assert compute_percentile_interval(np.arange(101.0), 0.9) == pytest.approx((5.0, 95.0))
    rng = np.random.default_rng(RNG_SEED)
    y = rng.normal(500, 50, 2000)
    stats = {"mean": calculate_mean_target}
    first = bootstrap_statistics(y, y, stats, 200, seed=1)["mean"]
    second = bootstrap_statistics(y, y, stats, 200, seed=1)["mean"]
    assert np.array_equal(first, second)
    low, high = compute_percentile_interval(first, 0.95)
    assert low < y.mean() < high
    assert high - low < 10  # erro-padrão ≈ 1,1 → IC ≈ 4,4


def test_bootstrap_metric_difference_sign():
    """Um modelo perfeito tem RMSE menor que um ruidoso em todas as réplicas."""
    rng = np.random.default_rng(RNG_SEED)
    y = rng.normal(0, 1, 300)
    noisy = y + rng.normal(0, 1, 300)
    differences = bootstrap_metric_difference(y, y, noisy, calculate_rmse, 50, seed=0)
    assert (differences < 0).all()


# --- Comparação -------------------------------------------------------------------------------
def test_corrected_ttest_matches_manual_formula():
    """A estatística segue a fórmula de Nadeau & Bengio."""
    differences = [1.0, 1.2, 0.8, 1.1, 0.9]
    result = corrected_resampled_ttest(differences, n_train=800, n_test=200)
    variance = np.var(differences, ddof=1)
    expected = np.mean(differences) / np.sqrt((1 / 5 + 200 / 800) * variance)
    assert result.statistic == pytest.approx(expected)
    assert result.degrees_of_freedom == 4
    assert 0.0 < result.p_value < 0.05


def test_corrected_ttest_edge_cases():
    """Variância zero e poucas dobras são tratadas explicitamente."""
    assert corrected_resampled_ttest([0.0, 0.0, 0.0], 80, 20).p_value == 1.0
    constant = corrected_resampled_ttest([2.0, 2.0, 2.0], 80, 20)
    assert constant.statistic == np.inf
    assert constant.p_value == 0.0
    with pytest.raises(ValueError, match="ao menos 2"):
        corrected_resampled_ttest([1.0], 80, 20)


# --- Validação cruzada ------------------------------------------------------------------------
def test_cross_validation_folds():
    """K dobras cobrem todas as linhas, com métricas finitas e resultado reprodutível."""
    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([3.0, -2.0, 1.0]) + rng.normal(0, 0.1, 200)
    results = cross_validate_model(LinearRegression, X, y, n_splits=4, seed=RNG_SEED)
    assert len(results) == 4
    assert sum(fold.n_test for fold in results) == 200
    assert all(fold.r2 > 0.99 for fold in results)
    again = cross_validate_model(LinearRegression, X, y, n_splits=4, seed=RNG_SEED)
    assert [f.rmse for f in results] == [f.rmse for f in again]
    frame = results_to_frame("ols", results)
    assert frame.height == 4
    assert set(frame["model"]) == {"ols"}


# --- Equidade ---------------------------------------------------------------------------------
def make_fairness_problem():
    """Dois grupos grandes (um com viés de +10) e um grupo pequeno."""
    rng = np.random.default_rng(RNG_SEED)
    y_true = rng.normal(500, 80, 940)
    groups = np.array(["A"] * 450 + ["B"] * 450 + ["C"] * 40)
    y_pred = y_true + rng.normal(0, 5, 940)
    y_pred[groups == "A"] += 10.0
    return y_true, y_pred, groups


@pytest.mark.smoke
def test_group_equity_detects_bias_gap():
    """O viés de +10 em A aparece como diferença significativa vs. B; C é omitido."""
    y_true, y_pred, groups = make_fairness_problem()
    metrics, gaps = analyze_group_equity(y_true, y_pred, groups, "grupo", "B", 200, 0.95, 100, 0)
    assert set(metrics["group"]) == {"A", "B"}
    bias = metrics.filter(pl.col("group") == "A")["mean_error"][0]
    assert bias == pytest.approx(10.0, abs=1.5)
    row = gaps.filter((pl.col("group") == "A") & (pl.col("statistic") == "mean_error"))
    assert row["gap"][0] == pytest.approx(10.0, abs=1.5)
    assert row["ci_low"][0] > 0
    assert row["significant"][0]
    assert (metrics["mae_low"] <= metrics["mae"]).all()
    assert (metrics["mae"] <= metrics["mae_high"]).all()


def test_group_equity_missing_reference_returns_empty_gaps():
    """Sem o grupo de referência não há diferenças, mas as métricas são devolvidas."""
    y_true, y_pred, groups = make_fairness_problem()
    metrics, gaps = analyze_group_equity(y_true, y_pred, groups, "g", "Z", 50, 0.95, 100, 0)
    assert metrics.height == 2
    assert gaps.is_empty()
    assert "significant" in gaps.columns


# --- SHAP -------------------------------------------------------------------------------------
def test_shap_global_and_group_summaries():
    """A feature que gera o alvo domina o SHAP global e os resumos por grupo."""
    rng = np.random.default_rng(RNG_SEED)
    X = rng.integers(0, 10, size=(300, 4)).astype(np.float32)
    y = 15.0 * X[:, 0] + rng.normal(0, 1.0, 300)
    model = build_gradient_boosting(
        GradientBoostingSettings(n_estimators=40, max_depth=3, n_jobs=1, min_child_weight=1.0),
        seed=RNG_SEED,
    )
    model.fit(X, y)
    values = compute_shap_values(model, X)
    assert values.shape == (300, 4)
    names = ["Q001", "Q002", "Q003", "Q004"]
    importance = summarize_global_importance(values, names)
    assert importance["feature"][0] == "Q001"
    assert importance["label"][0] == "Escolaridade do pai"

    groups = np.array(["g1"] * 200 + ["g2"] * 80 + ["pequeno"] * 20)
    by_group = summarize_shap_by_group(values, names, groups, "atributo", min_group_size=50)
    assert set(by_group["group"]) == {"g1", "g2"}
    assert by_group.columns[:3] == ["attribute", "group", "n"]
    empty = summarize_shap_by_group(values, names, groups, "atributo", min_group_size=1000)
    assert empty.is_empty()
