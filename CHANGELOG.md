# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o
projeto adota [Versionamento Semântico](https://semver.org/lang/pt-BR/). O arquivo
é atualizado por `cz bump --changelog` (Commitizen) a cada release.

## [Não lançado]

### Adicionado

- Ingestão idempotente dos microdados do ENEM (INEP) com cache local, manifest SHA-256
  e leitura lazy (`pl.scan_csv`) gravada em Parquet particionado por ano.
- Preparação silver → gold com tratamento de presença, ausentes e proteção contra vazamento.
- Contratos de dados (Pandera) para as camadas silver e gold.
- Regressão linear de referência (statsmodels) e gradient boosting (XGBoost) com validação cruzada,
  teste t reamostrado corrigido e intervalos de confiança por bootstrap.
- Explicabilidade com SHAP (global e por grupo).
- Análise de equidade por renda, raça/cor, tipo de escola e região, com intervalos de confiança.
- CI (lint, tipos, segurança, complexidade, testes), pre-commit, documentação MkDocs e Makefile.
