"""Gráficos de comparação de modelos, equidade e explicabilidade."""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import shap
from matplotlib.figure import Figure

from visualization.theme import HEATMAP_CMAP, PALETTE

logger = logging.getLogger(__name__)

STATISTIC_TITLES: dict[str, str] = {
    "mean_score": "Nota média observada",
    "mean_error": "Viés do modelo (previsto - observado)",
    "mae": "Erro absoluto médio (MAE)",
    "rmse": "RMSE",
}


def save_figure(fig: Figure, name: str, figures_dir: Path) -> list[Path]:
    """Salva a figura em PNG (300 dpi) e SVG e a fecha.

    Parameters
    ----------
    fig : Figure
        Figura do matplotlib.
    name : str
        Nome base do arquivo (sem extensão).
    figures_dir : Path
        Pasta de destino (criada se necessário).

    Returns
    -------
    list[Path]
        Caminhos dos arquivos ``.png`` e ``.svg``.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [figures_dir / f"{name}.png", figures_dir / f"{name}.svg"]
    fig.savefig(paths[0], dpi=300)
    fig.savefig(paths[1])
    plt.close(fig)
    logger.info("Figura salva: %s", paths[0])
    return paths


def plot_model_comparison(test_metrics: pl.DataFrame, metric: str = "rmse") -> Figure:
    """Compara os modelos em uma métrica do conjunto de teste, com IC.

    Parameters
    ----------
    test_metrics : pl.DataFrame
        Colunas ``model``, ``<metric>``, ``<metric>_low`` e ``<metric>_high``.
    metric : str, optional
        Métrica exibida, by default "rmse".

    Returns
    -------
    Figure
        Figura com um ponto por modelo e barras de erro (IC).
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    values = test_metrics[metric].to_numpy()
    lower = values - test_metrics[f"{metric}_low"].to_numpy()
    upper = test_metrics[f"{metric}_high"].to_numpy() - values
    ax.errorbar(
        values,
        test_metrics["model"].to_list(),
        xerr=[lower, upper],
        fmt="o",
        color=PALETTE[0],
        capsize=4,
    )
    ax.set_xlabel(f"{metric.upper()} no teste (pontos de nota, IC 95%)")
    ax.set_ylabel("Modelo")
    ax.set_title("Desempenho dos modelos no conjunto de teste")
    return fig


def plot_group_estimates(
    metrics: pl.DataFrame,
    attribute: str,
    statistics: Sequence[str] = ("mean_score", "mean_error", "mae"),
) -> Figure:
    """Gráfico de floresta: estatística por grupo com IC de 95% (um painel por estatística).

    Parameters
    ----------
    metrics : pl.DataFrame
        Saída de ``analyze_group_equity`` (``metrics``) para um atributo.
    attribute : str
        Nome do atributo (usado no título).
    statistics : Sequence[str], optional
        Estatísticas exibidas, uma por painel.

    Returns
    -------
    Figure
        Figura com os grupos ordenados pela nota média observada.
    """
    ordered = metrics.sort("mean_score")
    groups = ordered["group"].to_list()
    positions = np.arange(len(groups))
    fig, axes = plt.subplots(
        1, len(statistics), figsize=(4.2 * len(statistics), 0.7 * len(groups) + 2), sharey=True
    )
    for ax, statistic in zip(np.atleast_1d(axes), statistics, strict=True):
        values = ordered[statistic].to_numpy()
        lower = values - ordered[f"{statistic}_low"].to_numpy()
        upper = ordered[f"{statistic}_high"].to_numpy() - values
        ax.errorbar(values, positions, xerr=[lower, upper], fmt="o", color=PALETTE[0], capsize=3)
        if statistic == "mean_error":
            ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(positions, labels=groups)
        ax.set_xlabel("Pontos de nota")
        ax.set_title(STATISTIC_TITLES.get(statistic, statistic), fontsize=10)
    fig.suptitle(f"Equidade por {attribute} (IC 95% por bootstrap)")
    fig.tight_layout()
    return fig


def plot_shap_importance(importance: pl.DataFrame, top_n: int = 15) -> Figure:
    """Barras horizontais da importância global (média de |SHAP|).

    Parameters
    ----------
    importance : pl.DataFrame
        Saída de ``summarize_global_importance``.
    top_n : int, optional
        Quantidade de features exibidas, by default 15.

    Returns
    -------
    Figure
        Figura com as ``top_n`` features mais importantes.
    """
    top = importance.head(top_n).reverse()
    fig, ax = plt.subplots(figsize=(7, 0.4 * top.height + 1.5))
    ax.barh(top["label"].to_list(), top["mean_abs_shap"].to_numpy(), color=PALETTE[0])
    ax.set_xlabel("Média de |SHAP| (pontos de nota)")
    ax.set_title("Importância global das variáveis socioeconômicas")
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,  # noqa: N803
    feature_labels: Sequence[str],
    max_display: int = 15,
) -> Figure:
    """Gráfico beeswarm do SHAP (distribuição dos efeitos e sentido por feature).

    Parameters
    ----------
    shap_values : np.ndarray
        Matriz ``(n, p)`` de valores SHAP.
    X : np.ndarray
        Matriz de features correspondente (códigos ordinais).
    feature_labels : Sequence[str]
        Rótulos legíveis das ``p`` features.
    max_display : int, optional
        Quantidade de features exibidas, by default 15.

    Returns
    -------
    Figure
        Figura do ``shap.summary_plot``.
    """
    plt.figure()  # o shap desenha na figura corrente: garante uma figura nova
    shap.summary_plot(
        shap_values, X, feature_names=list(feature_labels), max_display=max_display, show=False
    )
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Valor SHAP (impacto na nota, em pontos)")
    return fig


def plot_shap_group_heatmap(by_group: pl.DataFrame, attribute: str, top_n: int = 12) -> Figure:
    """Mapa de calor da importância SHAP (média de |SHAP|) por grupo.

    Parameters
    ----------
    by_group : pl.DataFrame
        Saída de ``summarize_shap_by_group`` para um atributo.
    attribute : str
        Nome do atributo (usado no título).
    top_n : int, optional
        Quantidade de variáveis exibidas (as mais importantes no geral), by default 12.

    Returns
    -------
    Figure
        Figura com variáveis nas linhas e grupos nas colunas.
    """
    overall = by_group.group_by("label").agg(pl.col("mean_abs_shap").mean()).sort("mean_abs_shap")
    top_labels = overall.tail(top_n)["label"].to_list()[::-1]
    pivot = (
        by_group.filter(pl.col("label").is_in(top_labels))
        .pivot(on="group", index="label", values="mean_abs_shap")
        .sort(pl.col("label").replace_strict({k: i for i, k in enumerate(top_labels)}))
    )
    fig, ax = plt.subplots(figsize=(1.4 * (pivot.width - 1) + 3, 0.45 * pivot.height + 2))
    sns.heatmap(
        pivot.drop("label").to_numpy(),
        annot=True,
        fmt=".1f",
        cmap=HEATMAP_CMAP,
        xticklabels=pivot.columns[1:],
        yticklabels=pivot["label"].to_list(),
        cbar_kws={"label": "Média de |SHAP| (pontos de nota)"},
        ax=ax,
    )
    ax.set_title(f"Importância SHAP por {attribute}")
    ax.set_xlabel("Grupo")
    ax.set_ylabel("Variável")
    fig.tight_layout()
    return fig
