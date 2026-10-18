# Model Card — Nota do ENEM por perfil socioeconômico

> Valores preenchidos a partir de `reports/tables/` (execução de referência: ENEM 2022 e 2023,
> semente 42, SHA-256 da gold `9dedd4ec…cf4d`). Regenere-os se o pipeline for reexecutado.

## Detalhes

- **Modelos:** regressão linear (OLS, statsmodels) como referência e gradient boosting (XGBoost).
- **Alvo:** `mean_score`, média simples das 4 provas objetivas e da redação (0–1000).
- **Entradas:** apenas o questionário socioeconômico `Q001`–`Q025`.
- **Versão/dados:** ver `reports/tables/model_comparison.json` (`gold_sha256`, semente).

## Uso pretendido

Análise agregada e educacional de como o perfil socioeconômico se associa à nota, e de como o
erro do modelo varia entre grupos.

## Fora de escopo

Decisões sobre pessoas (seleção, bolsas, ranqueamento), previsão individual de desempenho,
inferência causal e qualquer uso com dados de outras edições sem revalidação.

## Dados de treino

ENEM (INEP), edições configuradas em `configs/config.yaml`, apenas participantes presentes nas
quatro provas, com redação válida e não treineiros. Ver `reports/datasheets/datasheet_enem.md`.

## Avaliação

| Modelo | RMSE teste (IC 95%) | MAE teste (IC 95%) | R² teste (IC 95%) |
|---|---|---|---|
| Regressão linear | 71,59 [71,24; 71,91] | 56,91 [56,60; 57,18] | 0,319 [0,314; 0,324] |
| Gradient boosting | 70,98 [70,62; 71,31] | 56,41 [56,10; 56,67] | 0,330 [0,325; 0,335] |

Treino: 400 mil linhas; teste: 100 mil (amostra reprodutível de 3.945.675 candidatos). A validação
cruzada (5 folds) concorda: R² 0,327 ± 0,003 para o boosting e 0,318 ± 0,003 para o OLS.

Comparação: diferença de RMSE (linear − boosting) = 0,61 pontos, IC 95% bootstrap pareado
[0,55; 0,66]; teste t reamostrado corrigido na CV, t = 30,8, p = 6,6·10⁻⁶ (4 g.l.). A diferença é
estatisticamente significativa, porém pequena na prática. Hiperparâmetros fixos, sem tuning.

## Avaliação por fatia e equidade

Gradient boosting no conjunto de teste. Viés = previsto − observado (pontos); IC 95% bootstrap.
Fatias com menos de 100 linhas seriam omitidas (`fairness.min_group_size`); nenhuma foi omitida.

| Eixo | Grupo | n | Viés | MAE | R² |
|---|---|---|---|---|---|
| Renda | Até 1 SM | 34.666 | −0,0 [−0,7; 0,7] | 53,6 | 0,135 |
| Renda | 1 a 3 SM | 41.630 | 0,3 [−0,4; 0,9] | 56,6 | 0,159 |
| Renda | 3 a 6 SM | 13.715 | 1,1 [−0,1; 2,4] | 60,4 | 0,132 |
| Renda | Mais de 6 SM | 9.989 | −0,9 [−2,3; 0,6] | 60,0 | 0,134 |
| Raça/cor | Branca | 41.867 | −6,9 [−7,6; −6,2] | 58,3 | 0,298 |
| Raça/cor | Parda | 41.965 | 4,4 [3,7; 5,1] | 55,3 | 0,264 |
| Raça/cor | Preta | 12.380 | 8,8 [7,6; 9,9] | 53,6 | 0,195 |
| Raça/cor | Amarela | 1.747 | 3,7 [0,5; 7,1] | 56,3 | 0,397 |
| Raça/cor | Indígena | 517 | 25,6 [19,8; 31,2] | 55,3 | 0,158 |
| Raça/cor | Não declarada | 1.524 | −5,4 [−9,1; −1,9] | 59,3 | 0,348 |
| Escola | Privada | 10.425 | −5,4 [−6,9; −4,0] | 58,6 | 0,089 |
| Escola | Pública | 37.786 | 9,6 [8,9; 10,3] | 54,9 | 0,165 |
| Escola | Não respondeu | 51.789 | −5,6 [−6,2; −5,0] | 57,0 | 0,350 |
| Região | Sudeste | 34.114 | −2,8 [−3,5; −2,1] | 56,0 | 0,301 |
| Região | Nordeste | 36.010 | −4,0 [−4,7; −3,3] | 57,6 | 0,310 |
| Região | Norte | 10.694 | 8,9 [7,7; 10,1] | 54,0 | 0,283 |
| Região | Centro-Oeste | 8.224 | 10,2 [8,6; 11,8] | 58,3 | 0,319 |
| Região | Sul | 10.958 | 6,9 [5,6; 8,1] | 54,5 | 0,297 |

Achados (diferenças de viés vs. grupo de referência, em `fairness_gaps.csv`):

- **Renda:** nenhum IC de viés exclui zero; sem disparidade sistemática de viés por faixa. O MAE é
  um pouco maior nas faixas mais altas (≈ 60 vs. 53,6 em "Até 1 SM").
- **Raça/cor:** o modelo superestima a nota de indígenas (+32,5 vs. brancos, IC [26,6; 38,1]),
  pretos (+15,7 [14,4; 17,1]), pardos (+11,3 [10,3; 12,3]) e amarelos (+10,6 [7,2; 14,1]).
  Indígenas têm só 517 casos: IC largo.
- **Escola:** pública superestimada em +15,0 [13,4; 16,6] vs. privada. "Não respondeu" não difere
  significativamente da privada (−0,2 [−1,7; 1,4]).
- **Região:** Centro-Oeste (+13,0), Norte (+11,7) e Sul (+9,7) superestimados vs. Sudeste;
  Nordeste praticamente igual (−1,2 [−2,2; −0,1]).
- **Leitura:** o perfil do questionário "promete" mais nota do que esses grupos obtêm; a diferença
  vem de fatores não observados (qualidade da escola, território), não é causal.
- O R² é baixo dentro de cada faixa de renda (0,13–0,16): a renda explica a maior parte da
  variação *entre* faixas, pouco dentro delas.
- Não foi aplicada mitigação; as disparidades são apenas reportadas.

## Limitações conhecidas

- Seleção: só entram quem compareceu; faltosos (mais frequentes entre os mais pobres) ficam de fora.
- Variáveis autodeclaradas; `Q001`/`Q002` têm categoria "não sei".
- Raça/cor, escola e região não são features, mas as demais variáveis podem ser proxies.
- Associação não é causalidade; SHAP descreve o modelo, não o mundo.

## Considerações éticas

O modelo reproduz a desigualdade presente nos dados. Reportar diferenças de erro entre grupos
**não** as justifica nem as corrige; a análise serve para dar visibilidade, não para classificar pessoas.
