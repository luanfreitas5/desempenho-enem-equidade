# Atalhos reprodutíveis. `make check` replica o CI localmente.
# 'src' vira a raiz do path ao rodar o script.
RUN := uv run python src/main.py

# PYTHONHASHSEED precisa ser exportado ANTES do interpretador iniciar: definido
# dentro do processo, não afeta a ordem de iteração de conjuntos já criada.
export PYTHONHASHSEED := 42

.DEFAULT_GOAL := help
.PHONY: help install lock export update hooks pre-commit 	lint format typecheck security quality check 	test smoke test-all coverage 	ingest prepare train explain fairness pipeline reproduce 	docs docs-serve docs-deploy release profile clean

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente --------------------------------------------------------------
install:  ## Instala todas as dependências do uv.lock e os hooks do pre-commit
	uv sync --locked --all-groups
	uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

lock:  ## Gera/atualiza o uv.lock (commitar)
	uv lock

export:  ## Exporta requirements.txt para quem usa pip
	uv export --no-hashes -o requirements.txt

update:  ## Atualiza todas as dependências e sincroniza
	uv lock --upgrade
	uv sync --all-groups

hooks:  ## Gera o baseline do detect-secrets (hooks já instalados por `make install`)
	uv run detect-secrets scan > .secrets.baseline

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	uv run pre-commit run --all-files

# --- Pipeline --------------------------------------------------------------
ingest:  ## Etapa 1: baixa os microdados (com cache) e grava o Parquet silver por ano
	$(RUN) ingest

prepare:  ## Etapa 2: silver -> gold (presença, alvo, grupos, ausentes)
	$(RUN) prepare

train:  ## Etapa 3: regressão linear (referência) e gradient boosting com CV
	$(RUN) train

explain:  ## Etapa 4: SHAP global e por grupo
	$(RUN) explain

fairness:  ## Etapa 5: análise de equidade com intervalos de confiança
	$(RUN) fairness

pipeline:  ## Executa todas as etapas, da ingestão à equidade
	$(RUN) all

reproduce: pipeline  ## Reproduz todos os resultados (sem DVC: o pipeline é idempotente)

# --- Qualidade -------------------------------------------------------------
lint:  ## Verifica lint e formatação (ruff), sem alterar arquivos
	uv run ruff check src tests
	uv run ruff format --check src tests

format:  ## Corrige lint e formata o código (ruff)
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:  ## Type checking estático (basedpyright)
	uv run basedpyright

security:  ## Segurança do código e das dependências (bandit + pip-audit)
	uv run bandit -c pyproject.toml -r src
	uv run pip-audit

quality:  ## Código morto, modernização, complexidade e docstrings
	uv run vulture
	uv run refurb src
	uv run xenon --max-absolute B --max-modules A --max-average A src
	uv run interrogate src

check: lint typecheck security quality test  ## Roda localmente tudo o que o CI roda

# --- Testes ----------------------------------------------------------------
test:  ## Roda os testes com cobertura (sem os lentos)
	uv run pytest -m "not slow" --cov-report=term-missing

smoke:  ## Roda apenas os smoke tests (os mesmos do pre-commit)
	uv run pytest -m smoke -q --no-cov

test-all:  ## Roda a suíte completa, inclusive os testes lentos
	uv run pytest

coverage:  ## Gera o relatório de cobertura em HTML
	uv run pytest -m "not slow" --cov-report=html
	@echo "Relatório disponível em htmlcov/index.html"

# --- Documentação e release ------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	uv run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	uv run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	uv run mkdocs gh-deploy --force

release:  ## Cria uma nova release (versão + changelog + tag)
	uv run cz bump --changelog

# --- Utilitários -----------------------------------------------------------
profile:  ## Profiling do pipeline completo com scalene
	uv run scalene src/main.py all

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml site
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
