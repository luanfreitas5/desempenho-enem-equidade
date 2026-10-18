"""Nomes de colunas dos microdados do ENEM e grupos de variáveis.

As colunas do INEP mantêm o nome original (``NU_NOTA_CN``, ``Q006`` ...); as
colunas derivadas pelo projeto usam nomes em inglês (``mean_score``,
``income_group`` ...).
"""

from typing import Final

# --- Provas objetivas e redação (origem do alvo; NUNCA usadas como feature) ---
SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
)
ESSAY_SCORE_COLUMN: Final[str] = "NU_NOTA_REDACAO"
ESSAY_STATUS_COLUMN: Final[str] = "TP_STATUS_REDACAO"
PRESENCE_COLUMNS: Final[tuple[str, ...]] = (
    "TP_PRESENCA_CN",
    "TP_PRESENCA_CH",
    "TP_PRESENCA_LC",
    "TP_PRESENCA_MT",
)

# --- Atributos de perfil usados na análise de equidade ------------------------
RACE_COLUMN: Final[str] = "TP_COR_RACA"
SCHOOL_TYPE_COLUMN: Final[str] = "TP_ESCOLA"
STATE_COLUMN: Final[str] = "SG_UF_PROVA"
TRAINEE_COLUMN: Final[str] = "IN_TREINEIRO"
INCOME_COLUMN: Final[str] = "Q006"

# --- Questionário socioeconômico (Q001 a Q025) --------------------------------
QUESTIONNAIRE_COLUMNS: Final[tuple[str, ...]] = tuple(f"Q{i:03d}" for i in range(1, 26))
# Questões cuja resposta é um número (e não uma letra de A a Q).
NUMERIC_QUESTIONS: Final[tuple[str, ...]] = ("Q005",)

# --- Colunas lidas do CSV bruto e persistidas na camada silver ----------------
INGEST_COLUMNS: Final[tuple[str, ...]] = (
    "NU_ANO",
    TRAINEE_COLUMN,
    RACE_COLUMN,
    SCHOOL_TYPE_COLUMN,
    STATE_COLUMN,
    *PRESENCE_COLUMNS,
    ESSAY_STATUS_COLUMN,
    *SCORE_COLUMNS,
    ESSAY_SCORE_COLUMN,
    *QUESTIONNAIRE_COLUMNS,
)
RAW_YEAR_COLUMN: Final[str] = "NU_ANO"
PARTITION_COLUMN: Final[str] = "ano"

# --- Colunas da camada gold ----------------------------------------------------
YEAR_COLUMN: Final[str] = "year"
TARGET_COLUMN: Final[str] = "mean_score"
INCOME_GROUP_COLUMN: Final[str] = "income_group"
RACE_GROUP_COLUMN: Final[str] = "race"
SCHOOL_GROUP_COLUMN: Final[str] = "school_type"
REGION_COLUMN: Final[str] = "region"
SENSITIVE_COLUMNS: Final[tuple[str, ...]] = (
    INCOME_GROUP_COLUMN,
    RACE_GROUP_COLUMN,
    SCHOOL_GROUP_COLUMN,
    REGION_COLUMN,
)
GOLD_COLUMNS: Final[tuple[str, ...]] = (
    YEAR_COLUMN,
    TARGET_COLUMN,
    *SENSITIVE_COLUMNS,
    *QUESTIONNAIRE_COLUMNS,
)

# --- Proteção contra vazamento do alvo ----------------------------------------
# Qualquer coluna com estes prefixos carrega informação (direta ou indireta)
# sobre o desempenho na prova e não pode virar feature.
LEAKAGE_PREFIXES: Final[tuple[str, ...]] = (
    "NU_NOTA_",
    "TX_RESPOSTAS_",
    "TX_GABARITO_",
    "CO_PROVA_",
    "TP_PRESENCA_",
    "TP_STATUS_REDACAO",
)
