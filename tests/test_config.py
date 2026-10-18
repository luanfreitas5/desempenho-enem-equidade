"""Testes de configuração, logging, ambiente e utilitários."""

import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from config.environment import seed_everything
from config.logging import configure_logging
from config.paths import CONFIG_DIR, ROOT, resolve_path
from config.settings import load_settings, read_yaml
from exceptions.configuration import ConfigurationError
from utils.hashing import hash_file
from utils.progress import create_progress
from utils.serialization import read_json, write_json, write_table


@pytest.mark.smoke
def test_load_settings_matches_version_and_defaults():
    """As configurações reais carregam e a versão do YAML coincide com o pacote."""
    settings = load_settings()
    package_init = (ROOT / "src" / "__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{settings.version}"' in package_init
    assert settings.project.random_seed == 42
    assert settings.ingestion.years
    assert settings.fairness.reference_groups["income_group"] == "Mais de 6 SM"


def test_load_settings_rejects_http_url(tmp_path: Path):
    """URL sem HTTPS deve falhar na inicialização com erro tipado."""
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, config_dir)
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace("https://", "http://"), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError):
        load_settings(config_dir)


def test_load_settings_rejects_unknown_field(tmp_path: Path):
    """Campos desconhecidos são proibidos (extra='forbid')."""
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, config_dir)
    with (config_dir / "config.yaml").open("a", encoding="utf-8") as handle:
        handle.write("\ncampo_inexistente: 1\n")
    with pytest.raises(ConfigurationError):
        load_settings(config_dir)


def test_read_yaml_errors(tmp_path: Path):
    """Arquivo ausente, YAML inválido e raiz não mapeada geram ConfigurationError."""
    with pytest.raises(ConfigurationError):
        read_yaml(tmp_path / "nao_existe.yaml")
    invalid = tmp_path / "invalido.yaml"
    invalid.write_text("a: [1, 2", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        read_yaml(invalid)
    not_mapping = tmp_path / "lista.yaml"
    not_mapping.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        read_yaml(not_mapping)


def test_resolve_path_relative_and_absolute(tmp_path: Path):
    """Caminhos relativos partem da raiz; absolutos permanecem."""
    assert resolve_path("data/bronze") == ROOT / "data" / "bronze"
    assert resolve_path(tmp_path) == tmp_path


def test_configure_logging_creates_daily_file(tmp_path: Path, restore_logging: None):
    """O logging grava em logs/log_YYYY-MM-DD.log e rejeita níveis inválidos."""
    configure_logging("INFO", tmp_path / "logs")
    logging.getLogger("teste").info("mensagem de teste")
    logging.shutdown()
    files = list((tmp_path / "logs").glob("log_*.log"))
    assert len(files) == 1
    with pytest.raises(ValueError, match="inválido"):
        configure_logging("NIVEL_FALSO")


def test_seed_everything_is_deterministic():
    """Duas chamadas com a mesma semente geram a mesma sequência."""
    seed_everything(7)
    first = np.random.rand(3)  # noqa: NPY002
    seed_everything(7)
    assert np.allclose(first, np.random.rand(3))  # noqa: NPY002


def test_hash_file_known_value(tmp_path: Path):
    """SHA-256 de 'abc' é um valor conhecido."""
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_file(path) == expected
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "ausente.txt")


def test_serialization_roundtrip(tmp_path: Path):
    """JSON e CSV são gravados criando a pasta de destino."""
    json_path = write_json({"a": 1, "nome": "ação"}, tmp_path / "sub" / "x.json")
    assert read_json(json_path) == {"a": 1, "nome": "ação"}
    csv_path = write_table(pl.DataFrame({"a": [1, 2]}), tmp_path / "sub" / "x.csv")
    assert pl.read_csv(csv_path).shape == (2, 1)


def test_create_progress_runs():
    """A barra de progresso padrão funciona como gerenciador de contexto."""
    with create_progress() as progress:
        task = progress.add_task("teste", total=2)
        progress.advance(task, 2)
        assert progress.tasks[0].completed == 2
