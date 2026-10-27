# Datasheet — Microdados do ENEM (INEP), edições 2022 e 2023

## 1. Motivação
- **Por que foi criado? Para qual tarefa?** Os microdados são divulgados pelo INEP para pesquisa e formulação de políticas educacionais. Neste projeto servem para estudar a relação entre perfil socioeconômico e nota, e a equidade do erro de um modelo preditivo.
- **Quem criou e quem financiou?** INEP (Ministério da Educação), financiamento público. O recorte e a análise deste repositório são de Luan Freitas, sem financiamento.

## 2. Composição
- **O que cada instância representa?** Um inscrito no ENEM em uma edição.
- **Número de instâncias e período coberto:** 7.410.060 inscritos (silver, edições 2022 e 2023); 3.945.675 após o funil de seleção (53,25%). Cada edição tem ~76 colunas.

  | Etapa do funil | Linhas | Retido |
  |---|---|---|
  | Inscritos (silver) | 7.410.060 | 100% |
  | Não treineiro | 6.277.261 | 84,71% |
  | Presente nas quatro provas objetivas | 4.094.596 | 55,26% |
  | Redação sem problemas | 3.945.675 | 53,25% |

  Fonte: `reports/tables/participation_funnel.csv`.
- **Campos, tipos e contrato:** `src/schemas/dataset.py` (Pandera, camadas silver e gold). Usados: presença, notas (4 provas + redação), situação da redação, treineiro, raça/cor, tipo de escola, UF de aplicação e questionário `Q001`–`Q025` (dicionário em `docs/dicionario_dados.md`).
- **Rótulo:** `mean_score`, média simples das 4 provas objetivas e da redação (0–1000), calculada na etapa gold; não existe no dado bruto.
- **Dados ausentes (taxa por coluna):** respostas ausentes do questionário viram a categoria `ND`. TODO(owner): tabela de taxa de ausência por coluna (hoje apenas no notebook `01_eda.ipynb`). `TP_ESCOLA` tem grande parcela de "Não respondeu" (51,8% do teste), pois a informação vem sobretudo de concluintes.
- **Campos sensíveis / PII e tratamento:** raça/cor é dado pessoal sensível (LGPD, art. 5º, II), usado apenas para agrupar a auditoria de equidade. Não são mantidos: número de inscrição, município/escola, respostas e gabaritos.
- **É amostra de uma população maior? Como amostrado?** Os microdados cobrem todos os inscritos. O treinamento usa amostra aleatória reprodutível (semente 42) de 500 mil linhas dos 3,95 milhões (400 mil treino / 100 mil teste), configurável em `configs/model_params.yaml`.

## 3. Processo de coleta
- **Fontes e sistemas de origem:** inscrição e aplicação das provas pelo INEP; questionário socioeconômico e raça/cor autodeclarados, sem verificação.
- **Período e frequência de coleta:** uma edição por ano (2022 e 2023 neste projeto).
- **Consentimento / aviso aos titulares:** coleta feita pelo INEP no ato da inscrição, com divulgação pública anonimizada nos termos da Lei de Acesso à Informação. O projeto não coleta dados novos dos titulares.

## 4. Pré-processamento
- **Limpeza, filtragem e linhas removidas por regra:** tipagem; remoção de treineiros (−1.132.799), ausentes em alguma prova objetiva (−2.182.665) e redação com problema (−148.921); categoria `ND` para ausentes do questionário; renda agrupada em faixas de salário mínimo. Detalhes em `docs/metodologia.md`.
- **Anonimização / pseudonimização:** base já anonimizada pelo INEP (sem nome ou CPF); o projeto descarta o identificador de inscrição e campos geográficos finos. Não há segredo ou *salt* envolvido.
- **Hash SHA-256 dos dados brutos:** registrado por arquivo no manifest gerado por `make ingest` (URL e SHA-256). Gold: `9dedd4ecbd0e764a211f896bfa24352b1fd49e5c3eaaa4e959fb01925627cf4d`. TODO(owner): copiar para cá o SHA-256 dos ZIPs brutos de 2022 e 2023 a partir do manifest.

## 5. Usos
- **Usos pretendidos:** pesquisa agregada sobre desigualdade educacional e auditoria de equidade de modelos.
- **Usos desaconselhados:** decisões individuais (admissão, bolsa, triagem), previsão individual de desempenho e inferência causal.
- **Riscos de viés conhecidos:** viés de seleção (quem falta às provas é mais frequente entre os mais pobres); autodeclaração; variáveis do questionário como *proxies* de renda e território; grupos pequenos (indígenas, n = 517 no teste) com intervalos largos.

## 6. Distribuição
- **Licença e restrições:** microdados abertos do INEP (Lei de Acesso à Informação); seguem os termos de uso do INEP. Este repositório **não** redistribui os dados.
- **Onde está armazenado:** cada pessoa baixa por `make ingest` para `data/bronze/` (fora do Git). Fonte: <https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem>. Sem DVC remote neste projeto.

## 7. Manutenção
- **Responsável:** Luan Freitas (luan.mgf@gmail.com).
- **Frequência de atualização:** a cada nova edição do INEP, ao adicionar o ano em `configs/config.yaml` e reexecutar `make pipeline`.
- **Como atender pedidos de exclusão de titulares:** o repositório não contém dados individuais (bronze, silver e gold ficam fora do Git, e nenhuma linha individual é publicada em figuras, tabelas ou README). Pedidos de exclusão devem ser dirigidos ao INEP, controlador dos dados; localmente, basta apagar `data/` e reexecutar.

## 8. LGPD
- **Base legal (art. 7 / art. 11):** o INEP é o controlador. Neste projeto, estudo por órgão de pesquisa com dados anonimizados (LGPD, arts. 7º, IV e 11, II, c). TODO(owner): confirmar juridicamente a base legal em uso institucional.
- **Prazo de retenção:** apenas local, enquanto o pesquisador mantiver `data/`; nada é publicado.
- **Relatório de impacto (RIPD) necessário?** Não para este uso (dados públicos anonimizados, resultados agregados com supressão de grupos com menos de `min_group_size` linhas). TODO(owner): reavaliar se houver cruzamento com outras bases ou uso institucional.
- **Risco de reidentificação:** combinação de quase-identificadores (UF + raça + renda + escola) em grupos pequenos; mitigado pela supressão de grupos pequenos e por não publicar linhas individuais.
