"""Testes de visualização, CLI e do pipeline ponta a ponta (dados sintéticos)."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import main as cli
from config.settings import Settings
from constants.artifacts import (
    GOLD_FILE,
    GRADIENT_BOOSTING_MODEL_NAME,
    LINEAR_MODEL_NAME,
    MODEL_FILES,
    TEST_PREDICTIONS_FILE,
)
from exceptions.data import DataDownloadError
from factories import build_raw_frame, write_microdata_zip
from fairness.group_metrics import analyze_group_equity
from pipelines.explainability import run_explainability
from pipelines.fairness import run_fairness
from pipelines.ingestion import run_ingestion
from pipelines.preparation import run_preparation
from pipelines.training import run_training
from pipelines.workflow import STAGE_NAMES, build_stage_runners
from visualization.plots import (
    plot_group_estimates,
    plot_model_comparison,
    plot_shap_beeswarm,
    plot_shap_group_heatmap,
    plot_shap_importance,
    save_figure,
)
from visualization.theme import apply_theme


def fake_fetch(url: str, destination: Path, timeout: int) -> None:
    """Simula o download do INEP gravando um ZIP sintético do ano da URL."""
    year = int(url.rsplit("_", 1)[1].split(".", maxsplit=1)[0])
    write_microdata_zip(destination, build_raw_frame(1200, year=year, seed=year), year)


def never_fetch(url: str, destination: Path, timeout: int) -> None:
    """Falha se for chamada: prova que o cache foi usado."""
    raise AssertionError("o cache deveria evitar o download")


# --- Visualização -----------------------------------------------------------------------------
def test_plots_render_and_save(tmp_path: Path):
    """Todas as figuras são geradas e salvas em PNG (300 dpi) e SVG."""
    apply_theme()
    rng = np.random.default_rng(0)
    y_true = rng.normal(500, 80, 400)
    groups = np.array(["Baixa", "Média", "Alta", "Muito alta"] * 100)
    metrics, _ = analyze_group_equity(
        y_true, y_true + rng.normal(0, 20, 400), groups, "income_group", "Alta", 30, 0.95, 20, 0
    )
    test_metrics = pl.DataFrame(
        {
            "model": ["linear", "boosting"],
            "rmse": [60.0, 55.0],
            "rmse_low": [58.0, 53.0],
            "rmse_high": [62.0, 57.0],
        }
    )
    importance = pl.DataFrame(
        {
            "feature": ["Q001", "Q002"],
            "label": ["Escolaridade do pai", "Escolaridade da mãe"],
            "mean_abs_shap": [10.0, 5.0],
            "mean_shap": [0.1, -0.1],
        }
    )
    by_group = pl.DataFrame(
        {
            "label": ["Escolaridade do pai", "Escolaridade da mãe"] * 2,
            "group": ["g1", "g1", "g2", "g2"],
            "mean_abs_shap": [10.0, 5.0, 8.0, 6.0],
        }
    )
    shap_values = rng.normal(size=(50, 2))
    figures = [
        plot_model_comparison(test_metrics),
        plot_group_estimates(metrics, "income_group"),
        plot_shap_importance(importance, 5),
        plot_shap_group_heatmap(by_group, "income_group", 5),
        plot_shap_beeswarm(shap_values, rng.normal(size=(50, 2)), ["A", "B"], 2),
    ]
    for index, figure in enumerate(figures):
        png, svg = save_figure(figure, f"fig_{index}", tmp_path / "figs")
        assert png.stat().st_size > 0
        assert svg.stat().st_size > 0


# --- CLI --------------------------------------------------------------------------------------
def test_cli_parser_and_exit_codes(
    pipeline_settings: Settings, monkeypatch: pytest.MonkeyPatch, restore_logging: None
):
    """O parser aceita as etapas; erros do projeto viram código de saída 1."""
    args = cli.build_parser().parse_args(["ingest", "--years", "2022", "2023", "--force"])
    assert (args.stage, args.years, args.force) == ("ingest", [2022, 2023], True)
    assert "all" in cli.build_parser().format_help()

    monkeypatch.setattr(cli, "load_settings", lambda _config_dir: pipeline_settings)
    monkeypatch.setattr(cli, "build_stage_runners", lambda *_args: {"train": lambda: None})
    assert cli.main(["train"]) == 0

    def failing() -> None:
        raise DataDownloadError("sem rede")

    monkeypatch.setattr(cli, "build_stage_runners", lambda *_args: {"train": failing})
    assert cli.main(["train"]) == 1


def test_stage_runners_cover_all_stages(pipeline_settings: Settings):
    """Há um executor para cada etapa declarada."""
    assert set(build_stage_runners(pipeline_settings, None, False)) == set(STAGE_NAMES)


# --- Pipeline ponta a ponta -------------------------------------------------------------------
@pytest.mark.integration
def test_full_pipeline_with_synthetic_data(pipeline_settings: Settings):
    """Ingestão → preparação → treino → SHAP → equidade, com dados sintéticos."""
    settings = pipeline_settings
    silver_files = run_ingestion(settings, fetch=fake_fetch)
    assert len(silver_files) == len(settings.ingestion.years)
    # Idempotência: com o silver pronto, nada é baixado de novo.
    assert run_ingestion(settings, fetch=never_fetch) == silver_files

    gold_path = run_preparation(settings)
    assert gold_path == settings.paths.resolved("gold_dir") / GOLD_FILE
    gold = pl.read_parquet(gold_path)
    assert gold.height > 1500
    assert (settings.paths.resolved("tables_dir") / "participation_funnel.csv").is_file()

    summary = run_training(settings)
    tables_dir = settings.paths.resolved("tables_dir")
    test_metrics = pl.read_csv(tables_dir / "test_metrics.csv")
    assert set(test_metrics["model"]) == {LINEAR_MODEL_NAME, GRADIENT_BOOSTING_MODEL_NAME}
    # Teste de regressão de métrica: a renda explica boa parte da nota sintética.
    assert (test_metrics["r2"] > 0.5).all()
    assert (test_metrics["rmse_low"] <= test_metrics["rmse_high"]).all()
    assert summary["comparison"]["cv_corrected_ttest"]["degrees_of_freedom"] == 2
    for model_file in MODEL_FILES.values():
        assert (settings.paths.resolved("models_dir") / model_file).is_file()
    predictions = pl.read_parquet(settings.paths.resolved("gold_dir") / TEST_PREDICTIONS_FILE)
    assert {"pred_linear_baseline", "pred_gradient_boosting"} <= set(predictions.columns)

    importance = run_explainability(settings)
    assert importance["feature"][0] == "Q006"  # renda é o sinal sintético
    assert (tables_dir / "shap_by_group.csv").is_file()

    metrics, gaps = run_fairness(settings)
    assert {"income_group", "race", "school_type", "region"} == set(metrics["attribute"])
    assert {"linear_baseline", "gradient_boosting"} == set(gaps["model"])
    assert (tables_dir / "fairness_gaps.csv").is_file()
    assert any((settings.paths.resolved("figures_dir")).glob("fairness_*.png"))
