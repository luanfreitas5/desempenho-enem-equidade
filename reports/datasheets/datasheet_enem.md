# Datasheet — Microdados do ENEM (INEP)

## Motivação

Dados públicos do Exame Nacional do Ensino Médio, divulgados pelo INEP para pesquisa e
formulação de políticas educacionais. Neste projeto são usados para estudar a relação entre
perfil socioeconômico e desempenho, e a equidade do erro de um modelo preditivo.

## Composição

- Uma linha por inscrito; cada edição tem milhões de linhas e ~76 colunas.
- Usadas aqui: presença, notas (4 provas + redação), situação da redação, treineiro, raça/cor,
  tipo de escola, UF de aplicação e questionário socioeconômico `Q001`–`Q025`
  (lista completa em `docs/dicionario_dados.md`).
- Não usadas e **não mantidas**: número de inscrição, município/escola, respostas e gabaritos.

## Coleta

Autodeclaração no ato da inscrição (questionário e raça/cor) e aplicação das provas pelo INEP.
O questionário é autodeclarado, sem verificação.

## Pré-processamento

Ver `docs/metodologia.md`: tipagem, filtro de presença e validade, categoria `ND` para ausentes,
agrupamento de renda em faixas de salário mínimo.

## Usos

Adequado para: pesquisa agregada sobre desigualdade educacional.
**Inadequado para:** decisões individuais (admissão, bolsa, triagem) ou inferência causal.

## Distribuição e licença

Os microdados são abertos (INEP, Lei de Acesso à Informação). Este repositório **não**
redistribui os dados: cada pessoa baixa por `make ingest`, que registra URL e SHA-256 no manifest.

## Privacidade e LGPD

- Base já anonimizada pelo INEP (sem nome/CPF). O projeto aplica **minimização**: descarta o
  identificador de inscrição e campos geográficos finos.
- Atributos como raça/cor são dados pessoais sensíveis (LGPD, art. 5º, II). O uso aqui é
  exclusivamente **agregado**, para auditoria de equidade (finalidade de estudo/pesquisa, cuja base
  legal deve ser confirmada juridicamente em uso institucional), com grupos pequenos suprimidos
  (`min_group_size`).
- Risco de reidentificação por combinação de quase-identificadores (UF + raça + renda + escola)
  em grupos pequenos: nenhuma linha individual é publicada em figuras, tabelas ou no README.
