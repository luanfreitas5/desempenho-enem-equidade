"""Ingestão e leitura de dados (camadas bronze, silver e gold).

Módulos
-------
downloader
    Download idempotente (com cache local e manifest) dos microdados do INEP.
reader
    Leitura lazy do CSV bruto com ``pl.scan_csv``.
writer
    Gravação de Parquet particionado por ano e de tabelas gold.
loader
    Leitura lazy das camadas silver e gold.
sampler
    Amostragem reprodutível de linhas.
splitter
    Divisão treino/teste reprodutível.
"""
