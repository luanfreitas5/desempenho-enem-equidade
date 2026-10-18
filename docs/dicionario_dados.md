# Dicionário de dados

Fonte: [Microdados do ENEM](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)
(INEP). Apenas as variáveis abaixo são lidas do CSV bruto; o identificador de inscrição
(`NU_INSCRICAO`) e todos os campos de texto livre/endereço **não** são mantidos
(minimização de dados, LGPD). Consulte o dicionário oficial do INEP (pasta `DICIONÁRIO` do ZIP)
para a definição completa de cada código.

## Variáveis lidas (camada silver)

| Variável | Descrição | Tipo na silver | Uso |
|---|---|---|---|
| `NU_ANO` | Ano da edição | Int16 | partição/validação |
| `IN_TREINEIRO` | 1 = treineiro | Int8 | filtro (excluídos por padrão) |
| `TP_COR_RACA` | Raça/cor: 0 não declarada, 1 branca, 2 preta, 3 parda, 4 amarela, 5 indígena | Int8 | **grupo de equidade** (não é feature) |
| `TP_ESCOLA` | Tipo de escola: 1 não respondeu, 2 pública, 3 privada | Int8 | **grupo de equidade** (não é feature) |
| `SG_UF_PROVA` | UF de aplicação da prova | texto | derivar **região** (grupo de equidade) |
| `TP_PRESENCA_CN/CH/LC/MT` | Presença: 0 faltou, 1 presente, 2 eliminado | Int8 | filtro (presente nas 4 provas) |
| `TP_STATUS_REDACAO` | Situação da redação (1 = sem problemas) | Int8 | filtro (apenas 1) |
| `NU_NOTA_CN/CH/LC/MT` | Notas de Ciências da Natureza, Humanas, Linguagens e Matemática | Float64 | **alvo** (nunca feature) |
| `NU_NOTA_REDACAO` | Nota da redação | Float64 | **alvo** (nunca feature) |
| `Q001`–`Q025` | Questionário socioeconômico (respostas A, B, C…; `Q005` é numérica) | texto | **features** |

## Questionário socioeconômico (features)

| Variável | Pergunta (resumo) |
|---|---|
| `Q001` / `Q002` | Escolaridade do pai / da mãe |
| `Q003` / `Q004` | Ocupação do pai / da mãe |
| `Q005` | Pessoas na residência (número) |
| `Q006` | **Renda familiar mensal** (A = nenhuma … Q = acima de 20 salários mínimos) |
| `Q007` | Empregado doméstico |
| `Q008`–`Q011` | Banheiros, quartos, carros, motocicletas |
| `Q012`–`Q018` | Geladeira, freezer, máquina de lavar roupa, de secar, micro-ondas, lavar louça, aspirador |
| `Q019`–`Q023` | TV em cores, DVD, TV por assinatura, celular, telefone fixo |
| `Q024` / `Q025` | Computador / acesso à Internet |

## Variáveis derivadas (camada gold)

| Variável | Definição |
|---|---|
| `year` | Ano da partição |
| `mean_score` | **Alvo.** Média simples das quatro provas objetivas e da redação (0–1000) |
| `income_group` | `Q006` em faixas: Até 1 SM (A–B), 1 a 3 SM (C–F), 3 a 6 SM (G–I), Mais de 6 SM (J–Q) |
| `race` | `TP_COR_RACA` rotulada |
| `school_type` | `TP_ESCOLA` rotulada |
| `region` | Região (Norte, Nordeste, Centro-Oeste, Sudeste, Sul) a partir da UF |

## Tratamento de ausentes

- **Questionário:** ausência vira a categoria explícita `ND` (não imputa a moda). No modelo linear é
  uma indicadora própria; no gradient boosting vira `NaN`, tratado nativamente pelas árvores.
- **Notas/presença:** só entram participantes presentes nas quatro provas, com redação sem
  problemas e todas as notas informadas (o funil é gravado em `reports/tables/participation_funnel.csv`).

## Variáveis que vazam a nota (proibidas como feature)

`NU_NOTA_*`, `NU_NOTA_COMP*`, `TX_RESPOSTAS_*`, `TX_GABARITO_*`, `CO_PROVA_*`, `TP_PRESENCA_*`,
`TP_STATUS_REDACAO` e o próprio alvo. A proteção é verificada em código
(`features.selection.validate_no_leakage`) e testada.
