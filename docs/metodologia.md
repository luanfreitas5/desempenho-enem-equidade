# Metodologia

## Camadas de dados (medalhão)

| Camada | Conteúdo | Local |
|---|---|---|
| bronze | ZIP/CSV do INEP exatamente como recebidos + `manifest.json` (URL, tamanho, SHA-256). Imutável | `data/bronze/raw/<ano>/` |
| silver | Colunas usadas, tipadas e validadas (Pandera), Parquet particionado por ano | `data/silver/enem/ano=YYYY/` |
| gold | Tabela analítica: alvo, grupos de equidade e questionário (sem nulos) | `data/gold/enem_gold.parquet` |

Dados nunca vão para o Git. A ingestão é idempotente: com a partição silver pronta nada é
baixado de novo; com o CSV em cache e manifest consistente o download é pulado.

## Leitura do CSV

O CSV do INEP usa `;` e codificação latin-1. O `pl.scan_csv` lazy só aceita UTF-8, então usamos
`encoding="utf8-lossy"`: bytes acentuados viram `�`, o que só afeta colunas de texto que o
projeto **não** usa (todas as colunas lidas são ASCII). A leitura é feita com
`infer_schema=False` (tudo texto) e a tipagem é explícita e validada.

## Modelagem

1. **Amostra** reprodutível (semente) das linhas da gold e divisão treino/teste (80/20).
2. **Referência:** regressão linear (OLS, statsmodels) com respostas do questionário em one-hot;
   intervalos dos coeficientes com erros-padrão robustos (HC1).
3. **Gradient boosting:** XGBoost sobre os códigos ordinais das respostas (`ND` = `NaN`).
4. **Validação cruzada K-fold** no treino: média ± IC 95% (t de Student) por modelo.
5. **Comparação:** teste t reamostrado corrigido (Nadeau & Bengio) sobre o RMSE por dobra e IC
   bootstrap pareado da diferença de RMSE no teste.
6. **Teste final:** RMSE, MAE e R² com IC bootstrap. O conjunto de teste não é usado para ajustar nada.

Métrica principal: **RMSE em pontos de nota** (mesma escala do ENEM, penaliza erros grandes);
MAE e R² complementam. Não há custo assimétrico de erro definido, portanto não se escolhe um limiar.

## Explicabilidade (SHAP)

TreeSHAP exato sobre uma amostra do **teste**. Resultados: importância global (média de |SHAP|),
beeswarm e importância por grupo (renda, raça/cor, escola, região) sob o mesmo modelo.
SHAP explica o *modelo*, não relações causais.

## Equidade

Para cada atributo sensível e cada modelo, no conjunto de teste:

- nota média observada, **viés** (previsto − observado), MAE, RMSE e R² por grupo, com IC bootstrap;
- diferenças *grupo − referência* (config `fairness.reference_groups`) em nota média, viés, MAE e RMSE,
  com IC bootstrap e indicador de significância (IC exclui zero);
- grupos com menos de `min_group_size` linhas são omitidos (IC instável).

Raça/cor, tipo de escola e região **não** são features: servem apenas para agrupar a análise.
Isso **não** garante equidade (as demais variáveis podem ser proxies deles) — por isso o erro
é medido por grupo.
