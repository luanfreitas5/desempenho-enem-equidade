# --- Configuração ----------------------------------------------------------
# 'src' vira a raiz do path ao rodar o script.
RUN := uv run python src/main.py

# PYTHONHASHSEED precisa ser exportado ANTES do interpretador iniciar: definido
# dentro do processo, não afeta a ordem de iteração de conjuntos já criada.
export PYTHONHASHSEED := 42

.DEFAULT_GOAL := help
.PHONY: help init venv install update lock export \
	lint typecheck security deadcode complexity docstrings modernize quality check \
	test smoke test-all coverage hooks pre-commit update-hooks release \
	ingest prepare train explain fairness pipeline \
	docs docs-serve docs-deploy profile clean cache jupyter add remove tree

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Ambiente --------------------------------------------------------------
init:  ## Inicializa o projeto (instala dependências + hooks)
	$(MAKE) install
	$(MAKE) hooks

venv:  ## Cria o ambiente virtual (requer: uv)
	uv venv

install:  ## Instala dependências (runtime + dev)
	uv sync --dev

update:  ## Atualiza todas as dependências e sincroniza
	uv lock --upgrade
	uv sync --all-groups

lock:  ## Gera/atualiza o uv.lock (commitar)
	uv lock

export:  ## Exporta requirements.txt para quem usa pip
	uv export --no-hashes -o requirements.txt

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

# --- Qualidade -------------------------------------------------------------
lint: ## Lint com ruff (Format + Check)
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Type checking estático (basedPyright)
	uv run basedpyright

security:  ## Análise de segurança (bandit + pip-audit)
	uv run bandit -r src -c pyproject.toml
	uv run pip-audit

deadcode:  ## Detecta código morto (vulture)
	uv run vulture src

complexity:  ## Limites de complexidade (xenon)
	uv run xenon --max-absolute B --max-modules A --max-average A src

docstrings:  ## Cobertura de docstrings (interrogate)
	uv run interrogate -v src

modernize:  ## Detecta código redundante (refurb)
	uv run refurb src

quality: lint typecheck security deadcode complexity docstrings modernize   ## Roda toda a suíte de qualidade (espelha o CI)

check: quality test  ## Roda localmente tudo o que o CI roda (qualidade + testes)

# --- Testes ----------------------------------------------------------------
test:  ## Roda os testes com cobertura
	uv run pytest -m "not slow"

smoke:  ## Roda apenas os smoke tests
	uv run pytest -m smoke -q --no-cov

test-all:  ## Roda a suíte completa, inclusive os testes lentos
	uv run pytest

coverage:  ## Gera o relatório de cobertura em HTML
	uv run pytest -m "not slow" --cov-report=html
	@echo "Relatório disponível em htmlcov/index.html"

hooks:  ## Instala os hooks do pre-commit e gera o baseline do detect-secrets
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	uv run detect-secrets scan > .secrets.baseline

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	uv run pre-commit run --all-files

update-hooks:  ## Atualiza os hooks do pre-commit
	uv run pre-commit autoupdate

release:  ## Cria uma nova release (versão + changelog + tag)
	uv run cz bump --changelog

# --- Documentação ----------------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	uv run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	uv run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	uv run mkdocs gh-deploy --force

# --- Utilitários -----------------------------------------------------------
profile:  ## Profiling do pipeline completo com scalene
	uv run scalene src/main.py all

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml site
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

cache:  ## Limpa o cache do uv
	uv cache clean

jupyter:  ## Abre o JupyterLab
	uv run jupyter lab

add:  ## Adiciona uma dependência: make add PKG=nome
	uv add $(PKG)

remove:  ## Remove uma dependência: make remove PKG=nome
	uv remove $(PKG)

tree:  ## Árvore de dependências
	uv tree
