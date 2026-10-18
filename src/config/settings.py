"""Carrega e valida as configurações de ``configs/*.yaml`` com Pydantic."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.paths import CONFIG_DIR, resolve_path
from exceptions.configuration import ConfigurationError

CONFIG_FILES: tuple[str, ...] = (
    "config.yaml",
    "paths.yaml",
    "model_params.yaml",
    "logging.yaml",
)


class _FrozenModel(BaseModel):
    """Base dos modelos de configuração: imutáveis e sem campos extras."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProjectSettings(_FrozenModel):
    """Metadados do projeto."""

    name: str
    random_seed: int = 42


class IngestionSettings(_FrozenModel):
    """Parâmetros de download e leitura dos microdados."""

    years: list[int] = Field(min_length=1)
    base_url: str
    separator: str = ";"
    # O ``scan_csv`` lazy do Polars só aceita UTF-8. Como as colunas usadas são
    # ASCII (códigos, notas e letras), ``utf8-lossy`` lê o arquivo latin-1 sem
    # perda nas colunas relevantes; apenas textos acentuados (não usados) mudam.
    encoding: Literal["utf8", "utf8-lossy"] = "utf8-lossy"
    keep_zip: bool = False
    timeout_seconds: int = Field(default=120, gt=0)

    @field_validator("base_url")
    @classmethod
    def _require_https_and_year_placeholder(cls, value: str) -> str:
        """Garante URL HTTPS com o marcador ``{year}``."""
        if not value.startswith("https://") or "{year}" not in value:
            raise ValueError("base_url deve ser HTTPS e conter o marcador {year}")
        return value


class PreparationSettings(_FrozenModel):
    """Regras de seleção de participantes na camada gold."""

    exclude_trainees: bool = True


class GradientBoostingSettings(_FrozenModel):
    """Hiperparâmetros do gradient boosting (XGBoost)."""

    n_estimators: int = Field(default=400, gt=0)
    max_depth: int = Field(default=6, gt=0)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    min_child_weight: float = Field(default=10.0, ge=0)
    n_jobs: int = -1


class ModelingSettings(_FrozenModel):
    """Amostragem, divisão dos dados e validação cruzada."""

    sample_size: int | None = Field(default=500_000, gt=0)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    n_splits: int = Field(default=5, ge=2)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    n_bootstrap: int = Field(default=500, ge=10)
    gradient_boosting: GradientBoostingSettings = GradientBoostingSettings()


class ExplainabilitySettings(_FrozenModel):
    """Parâmetros da análise SHAP."""

    sample_size: int = Field(default=20_000, gt=0)
    top_features: int = Field(default=15, gt=0)


class FairnessSettings(_FrozenModel):
    """Parâmetros da análise de equidade."""

    n_bootstrap: int = Field(default=1_000, ge=10)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    min_group_size: int = Field(default=100, gt=0)
    reference_groups: dict[str, str]


class PathsSettings(_FrozenModel):
    """Caminhos (relativos à raiz do projeto) de dados, modelos e relatórios."""

    raw_dir: Path
    silver_dir: Path
    gold_dir: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    logs_dir: Path

    def resolved(self, name: str) -> Path:
        """Devolve o caminho absoluto de um campo (ex.: ``"raw_dir"``).

        Parameters
        ----------
        name : str
            Nome do campo de caminho.

        Returns
        -------
        Path
            Caminho absoluto resolvido a partir da raiz do projeto.
        """
        return resolve_path(getattr(self, name))


class LoggingSettings(_FrozenModel):
    """Configuração do logger."""

    level: str = "INFO"
    backup_count: int = Field(default=30, ge=1)


class Settings(BaseSettings):
    """Configuração global do projeto, validada na inicialização.

    Variáveis de ambiente com prefixo ``ENEM_`` (ou um ``.env``) podem
    sobrescrever campos de primeiro nível.
    """

    model_config = SettingsConfigDict(
        env_prefix="ENEM_", env_file=".env", extra="forbid", frozen=True
    )

    version: str
    project: ProjectSettings
    ingestion: IngestionSettings
    preparation: PreparationSettings = PreparationSettings()
    modeling: ModelingSettings = ModelingSettings()
    explainability: ExplainabilitySettings = ExplainabilitySettings()
    fairness: FairnessSettings
    paths: PathsSettings
    logging: LoggingSettings = LoggingSettings()


def read_yaml(path: Path) -> dict[str, Any]:
    """Lê um arquivo YAML e devolve seu conteúdo como dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo YAML.

    Returns
    -------
    dict[str, Any]
        Conteúdo do arquivo (vazio se o arquivo estiver vazio).

    Raises
    ------
    ConfigurationError
        Se o arquivo não existir, for YAML inválido ou não for um mapeamento.
    """
    if not path.is_file():
        raise ConfigurationError(f"Arquivo de configuração não encontrado: {path}")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigurationError(f"YAML inválido em {path}: {error}") from error
    if not isinstance(content, dict):
        raise ConfigurationError(f"O arquivo {path} deve conter um mapeamento no nível raiz")
    return content


def load_settings(config_dir: Path | None = None) -> Settings:
    """Carrega todos os YAML de configuração e valida com Pydantic.

    Parameters
    ----------
    config_dir : Path or None, optional
        Pasta com os YAML. Se ``None``, usa ``configs/`` na raiz do projeto.

    Returns
    -------
    Settings
        Configuração validada e imutável.

    Raises
    ------
    ConfigurationError
        Se algum arquivo estiver ausente/malformado ou a validação falhar.

    Examples
    --------
    >>> settings = load_settings()
    >>> settings.project.random_seed
    42
    """
    directory = config_dir or CONFIG_DIR
    merged: dict[str, Any] = {}
    for file_name in CONFIG_FILES:
        merged.update(read_yaml(directory / file_name))
    try:
        return Settings(**merged)
    except ValueError as error:  # ValidationError herda de ValueError
        raise ConfigurationError(f"Configuração inválida: {error}") from error
