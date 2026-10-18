"""Rótulos legíveis e mapeamentos de categorias dos microdados do ENEM."""

from typing import Final

# Categoria explícita para respostas ausentes do questionário.
MISSING_CATEGORY: Final[str] = "ND"
UNKNOWN_GROUP: Final[str] = "Não informado"

# TP_COR_RACA (0 = não declarado).
RACE_LABELS: Final[dict[int, str]] = {
    0: "Não declarada",
    1: "Branca",
    2: "Preta",
    3: "Parda",
    4: "Amarela",
    5: "Indígena",
}

# TP_ESCOLA (1 = não respondeu; informação preenchida sobretudo por concluintes).
SCHOOL_TYPE_LABELS: Final[dict[int, str]] = {
    1: "Não respondeu",
    2: "Pública",
    3: "Privada",
}

# Q006 (renda familiar mensal) agrupada em salários mínimos (SM).
INCOME_GROUP_ORDER: Final[tuple[str, ...]] = (
    "Até 1 SM",
    "1 a 3 SM",
    "3 a 6 SM",
    "Mais de 6 SM",
)
INCOME_GROUP_BY_LETTER: Final[dict[str, str]] = {
    "A": "Até 1 SM",  # nenhuma renda
    "B": "Até 1 SM",
    "C": "1 a 3 SM",
    "D": "1 a 3 SM",
    "E": "1 a 3 SM",
    "F": "1 a 3 SM",
    "G": "3 a 6 SM",
    "H": "3 a 6 SM",
    "I": "3 a 6 SM",
    "J": "Mais de 6 SM",
    "K": "Mais de 6 SM",
    "L": "Mais de 6 SM",
    "M": "Mais de 6 SM",
    "N": "Mais de 6 SM",
    "O": "Mais de 6 SM",
    "P": "Mais de 6 SM",
    "Q": "Mais de 6 SM",
}

# Região geográfica a partir da UF de aplicação da prova.
REGION_BY_STATE: Final[dict[str, str]] = {
    "AC": "Norte",
    "AM": "Norte",
    "AP": "Norte",
    "PA": "Norte",
    "RO": "Norte",
    "RR": "Norte",
    "TO": "Norte",
    "AL": "Nordeste",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "MA": "Nordeste",
    "PB": "Nordeste",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste",
    "GO": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "MT": "Centro-Oeste",
    "ES": "Sudeste",
    "MG": "Sudeste",
    "RJ": "Sudeste",
    "SP": "Sudeste",
    "PR": "Sul",
    "RS": "Sul",
    "SC": "Sul",
}

# Rótulos curtos das 25 questões do questionário socioeconômico (para gráficos).
QUESTION_LABELS: Final[dict[str, str]] = {
    "Q001": "Escolaridade do pai",
    "Q002": "Escolaridade da mãe",
    "Q003": "Ocupação do pai",
    "Q004": "Ocupação da mãe",
    "Q005": "Pessoas na residência",
    "Q006": "Renda familiar",
    "Q007": "Empregado doméstico",
    "Q008": "Banheiros",
    "Q009": "Quartos",
    "Q010": "Carros",
    "Q011": "Motocicletas",
    "Q012": "Geladeira",
    "Q013": "Freezer",
    "Q014": "Máquina de lavar roupa",
    "Q015": "Máquina de secar roupa",
    "Q016": "Micro-ondas",
    "Q017": "Máquina de lavar louça",
    "Q018": "Aspirador de pó",
    "Q019": "TV em cores",
    "Q020": "DVD",
    "Q021": "TV por assinatura",
    "Q022": "Celular",
    "Q023": "Telefone fixo",
    "Q024": "Computador",
    "Q025": "Internet",
}
