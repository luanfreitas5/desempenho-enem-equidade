"""Etapas ponta a ponta do pipeline.

Módulos
-------
ingestion
    Etapa 1: download idempotente → CSV lazy → Parquet silver particionado por ano.
preparation
    Etapa 2: silver → gold (presença, alvo, grupos, ausentes) com contrato validado.
training
    Etapa 3: regressão linear (referência) e gradient boosting com validação cruzada.
explainability
    Etapa 4: SHAP global e por grupo.
fairness
    Etapa 5: equidade de desempenho e de erro, com intervalos de confiança.
workflow
    Encadeia todas as etapas (``run_all``).
"""
