"""Fixtures compartilhadas dos testes."""

import logging
from collections.abc import Iterator
from pathlib import Path

import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")  # backend sem janela, antes de qualquer import de pyplot

from config.settings import GradientBoostingSettings, PathsSettings, Settings, load_settings
from factories import build_raw_frame, build_silver_lazy
from features.engineering import build_gold_frame
from preprocessing.filtering import apply_filter_steps, build_filter_steps


@pytest.fixture
def raw_frame() -> pl.DataFrame:
    """DataFrame sintético no formato do CSV bruto."""
    return build_raw_frame()


@pytest.fixture
def silver_lazy() -> pl.LazyFrame:
    """Plano lazy silver sintético."""
    return build_silver_lazy()


@pytest.fixture
def gold_frame(silver_lazy: pl.LazyFrame) -> pl.DataFrame:
    """Tabela gold sintética (após filtros de presença)."""
    return build_gold_frame(apply_filter_steps(silver_lazy, build_filter_steps(True))).collect()


@pytest.fixture
def pipeline_settings(tmp_path: Path) -> Settings:
    """Configurações reais do projeto, com caminhos temporários e parâmetros mínimos."""
    base = load_settings()
    paths = PathsSettings(
        raw_dir=tmp_path / "bronze" / "raw",
        silver_dir=tmp_path / "silver" / "enem",
        gold_dir=tmp_path / "gold",
        models_dir=tmp_path / "models",
        figures_dir=tmp_path / "reports" / "figures",
        tables_dir=tmp_path / "reports" / "tables",
        logs_dir=tmp_path / "logs",
    )
    modeling = base.modeling.model_copy(
        update={
            "sample_size": None,
            "n_splits": 3,
            "n_bootstrap": 20,
            "gradient_boosting": GradientBoostingSettings(
                n_estimators=25, max_depth=3, n_jobs=1, min_child_weight=1.0
            ),
        }
    )
    return base.model_copy(
        update={
            "paths": paths,
            "modeling": modeling,
            "fairness": base.fairness.model_copy(update={"n_bootstrap": 30, "min_group_size": 10}),
            "explainability": base.explainability.model_copy(update={"sample_size": 300}),
        }
    )


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Restaura os handlers do logger raiz alterados por ``configure_logging``."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers[:]:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)
