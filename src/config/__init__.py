"""Configuração do projeto: settings validados, caminhos, logging e ambiente.

Módulos
-------
settings
    Carrega os YAML de ``configs/`` e valida com Pydantic (``load_settings``).
paths
    Caminhos centralizados com ``pathlib`` (``ROOT``, ``resolve_path``).
logging
    Logging com ``rich`` no console e arquivo diário (``configure_logging``).
environment
    Reprodutibilidade e sementes (``seed_everything``).
"""
