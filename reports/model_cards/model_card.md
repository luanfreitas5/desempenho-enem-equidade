# Model Card — Nota do ENEM por perfil socioeconômico v0.1.0

**Status:** DRAFT
**Bloqueios pendentes:**
- Responsável/contato e rastreabilidade por run de experimento (seção 1): bloqueiam qualquer aprovação formal.
- Validação externa/*out-of-time* (seções 6 e 10): bloqueia uso com edições posteriores a 2023.
- Plano de monitoramento (seção 10): bloqueia qualquer implantação (hoje o modelo só gera análise offline).

> Valores preenchidos a partir de `reports/tables/` (execução de referência: ENEM 2022 e 2023,
> semente 42, SHA-256 da gold `9dedd4ec…cf4d`). Regenere-os se o pipeline for reexecutado.

## 1. Detalhes do modelo
- **Responsável / contato:** Luan Freitas (luan.mgf@gmail.com); TODO(owner): canal de suporte caso haja uso institucional
- **Data:** 2026-10-18 (execução de referência)
- **Algoritmo:** referência — regressão linear OLS (statsmodels, erros robustos HC1); principal — gradient boosting (XGBoost), hiperparâmetros fixos em `configs/model_params.yaml`, sem tuning
- **Alvo:** `mean_score`, média simples das 4 provas objetivas e da redação (0–1000)
- **Entradas:** apenas o questionário socioeconômico `Q001`–`Q025`
- **Rastreabilidade:** Git SHA `2b2358b` (último commit antes desta documentação) · MLflow run: não se aplica (sem MLflow neste projeto; ver README, "Decisões técnicas") · hash dos dados `9dedd4ecbd0e764a211f896bfa24352b1fd49e5c3eaaa4e959fb01925627cf4d` (gold; `reports/tables/model_comparison.json`)
- **Licença:** MIT (código); microdados seguem os termos do INEP

## 2. Uso pretendido
- **Uso primário:** análise agregada e educacional de como o perfil socioeconômico se associa à nota e de como o erro do modelo varia entre grupos.
- **Usuários pretendidos:** pesquisadores, analistas de política educacional e estudantes de ciência de dados.
- **Humano no circuito:** sim — o score não alimenta nenhuma decisão automática; resultados são lidos por analistas, sempre com os intervalos de confiança.

## 3. Usos fora do escopo
- Decisões sobre pessoas: seleção, bolsas, triagem, ranqueamento ou qualquer negativa de acesso.
- Previsão individual de desempenho (R² de 0,33: o erro típico é ~57 pontos).
- Inferência causal (SHAP descreve o modelo, não o efeito de uma variável).
- Edições do ENEM fora de 2022–2023 sem revalidação (mudanças de prova, TRI ou perfil dos candidatos).

## 4. Fatores
- **Grupos relevantes:** faixa de renda, raça/cor, tipo de escola e região (UF de aplicação). Nenhum deles é feature; servem para fatiar a avaliação.
- **Ambientes:** microdados do INEP, edições 2022 e 2023; apenas candidatos presentes nas quatro provas, com redação válida e não treineiros.

## 5. Métricas
- **Métrica principal:** RMSE (pontos da nota, 0–1000) — penaliza mais os erros grandes, relevantes para uma análise de desigualdade em que os extremos importam; MAE e R² são reportados como apoio. Não há custo assimétrico de FP/FN: a tarefa é regressão e não dispara decisões.
- **Limiar de decisão:** não se aplica (regressão).
- **Calibração:** não se aplica a probabilidades; o viés (previsto − observado) por grupo substitui a checagem de calibração na seção 8. TODO(owner): curva de calibração previsto × observado por decil, se o modelo for usado fora de análise exploratória.

## 6. Dados de avaliação
Teste de 100 mil linhas, amostra reprodutível (semente 42) dos 3.945.675 candidatos que sobraram do funil (53,25% dos 7.410.060 inscritos); treino de 400 mil linhas. Validação cruzada em 5 folds sobre o treino. A divisão é aleatória, **não temporal**: não há avaliação *out-of-time*. Intervalos: bootstrap no teste e t de Student sobre os folds, ambos IC 95%.

## 7. Dados de treino
Microdados do ENEM 2022 e 2023 (INEP); resumo, composição e privacidade em [`reports/datasheets/datasheet_enem.md`](../datasheets/datasheet_enem.md).

## 8. Análise quantitativa

### Geral (teste, n = 100.000)

| Modelo | RMSE (IC 95%) | MAE (IC 95%) | R² (IC 95%) |
|---|---|---|---|
| Baseline: regressão linear (OLS) | 71,59 [71,24; 71,91] | 56,91 [56,60; 57,18] | 0,319 [0,314; 0,324] |
| Gradient boosting (XGBoost) | **70,98 [70,62; 71,31]** | **56,41 [56,10; 56,67]** | **0,330 [0,325; 0,335]** |

Validação cruzada (5 folds, IC 95% t): R² 0,327 ± 0,003 (boosting) e 0,318 ± 0,003 (OLS), de acordo com o teste.

Comparação: diferença de RMSE (OLS − boosting) = 0,61 pontos, IC 95% bootstrap pareado [0,55; 0,66]; teste t reamostrado corrigido na CV, t = 30,8, p = 6,6·10⁻⁶ (4 g.l.). A diferença é estatisticamente significativa, porém pequena na prática.

### Por fatia (boosting, teste)

Viés = previsto − observado (pontos); IC 95% bootstrap. Fatias com menos de 100 linhas seriam omitidas (`fairness.min_group_size`); nenhuma foi omitida. Baseline = MAE do OLS na mesma fatia.

| Recorte | n | Viés ± IC 95% | MAE (boosting) | MAE (baseline OLS) | R² |
|---|---|---|---|---|---|
| Renda: Até 1 SM | 34.666 | −0,0 [−0,7; 0,7] | 53,6 | 54,1 | 0,135 |
| Renda: 1 a 3 SM | 41.630 | 0,3 [−0,4; 0,9] | 56,6 | 57,0 | 0,159 |
| Renda: 3 a 6 SM | 13.715 | 1,1 [−0,1; 2,4] | 60,4 | 60,9 | 0,132 |
| Renda: Mais de 6 SM | 9.989 | −0,9 [−2,3; 0,6] | 60,0 | 60,6 | 0,134 |
| Raça/cor: Branca | 41.867 | −6,9 [−7,6; −6,2] | 58,3 | 58,8 | 0,298 |
| Raça/cor: Parda | 41.965 | 4,4 [3,7; 5,1] | 55,3 | 55,7 | 0,264 |
| Raça/cor: Preta | 12.380 | 8,8 [7,6; 9,9] | 53,6 | 54,1 | 0,195 |
| Raça/cor: Amarela | 1.747 | 3,7 [0,5; 7,1] | 56,3 | 57,0 | 0,397 |
| Raça/cor: Indígena | 517 | 25,6 [19,8; 31,2] | 55,3 | 56,3 | 0,158 |
| Raça/cor: Não declarada | 1.524 | −5,4 [−9,1; −1,9] | 59,3 | 60,0 | 0,348 |
| Escola: Privada | 10.425 | −5,4 [−6,9; −4,0] | 58,6 | 59,0 | 0,089 |
| Escola: Pública | 37.786 | 9,6 [8,9; 10,3] | 54,9 | 55,5 | 0,165 |
| Escola: Não respondeu | 51.789 | −5,6 [−6,2; −5,0] | 57,0 | 57,5 | 0,350 |
| Região: Sudeste | 34.114 | −2,8 [−3,5; −2,1] | 56,0 | 56,5 | 0,301 |
| Região: Nordeste | 36.010 | −4,0 [−4,7; −3,3] | 57,6 | 58,1 | 0,310 |
| Região: Norte | 10.694 | 8,9 [7,7; 10,1] | 54,0 | 54,5 | 0,283 |
| Região: Centro-Oeste | 8.224 | 10,2 [8,6; 11,8] | 58,3 | 59,0 | 0,319 |
| Região: Sul | 10.958 | 6,9 [5,6; 8,1] | 54,5 | 54,9 | 0,297 |

Pior fatia por MAE: renda de 3 a 6 SM (60,4). O R² é baixo dentro de cada faixa de renda (0,13–0,16): a renda explica a variação *entre* faixas, pouco *dentro* delas. Figuras: `reports/figures/fairness_gradient_boosting_*.png` e `shap_*.png`.

## 9. Considerações éticas e fairness
- **Atributos sensíveis auditados:** renda, raça/cor, tipo de escola e região (dados pessoais sensíveis em agregado; ver datasheet).
- **Disparidades encontradas** (diferença de viés vs. grupo de referência, IC 95% bootstrap, `reports/tables/fairness_gaps.csv`):
  - **Renda:** nenhum IC de viés exclui zero; sem disparidade sistemática de viés por faixa.
  - **Raça/cor:** o modelo superestima a nota de indígenas (+32,5 vs. brancos, IC [26,6; 38,1]; n = 517, IC largo), pretos (+15,7 [14,4; 17,1]), pardos (+11,3 [10,3; 12,3]) e amarelos (+10,6 [7,2; 14,1]).
  - **Escola:** pública superestimada em +15,0 [13,4; 16,6] vs. privada; "Não respondeu" não difere da privada (−0,2 [−1,7; 1,4]).
  - **Região:** Centro-Oeste (+13,0), Norte (+11,7) e Sul (+9,7) superestimados vs. Sudeste; Nordeste praticamente igual (−1,2 [−2,2; −0,1]).
  - **Leitura:** pelo questionário, esses grupos parecem ter perfil que "promete" mais nota do que obtêm; a diferença vem de fatores não observados (qualidade da escola, território), não é causal.
- **Mitigação aplicada:** nenhuma; as disparidades são apenas reportadas. TODO(owner): critério de aceitação (limiar de disparidade) se houver uso institucional.
- **Base legal LGPD:** dados públicos já anonimizados pelo INEP, uso exclusivamente agregado com supressão de grupos pequenos. Em uso institucional, a base legal deve ser confirmada juridicamente (ver datasheet, seção 8).
- O modelo reproduz a desigualdade presente nos dados. Reportar diferenças de erro entre grupos **não** as justifica nem as corrige; a análise dá visibilidade, não classifica pessoas.

## 10. Limitações, riscos e monitoramento
- **Limitações conhecidas:**
  - Viés de seleção: só entram quem compareceu às quatro provas (46,75% dos inscritos ficam de fora, mais frequentes entre os mais pobres).
  - Variáveis autodeclaradas; `Q001`/`Q002` têm categoria "não sei"; `TP_ESCOLA` tem muitos "Não respondeu".
  - Raça/cor, escola e região não são features, mas as demais variáveis podem ser *proxies*.
  - Hiperparâmetros fixos, sem tuning; treino em amostra de 500 mil linhas.
  - Associação não é causalidade.
- **Mudanças de distribuição previstas:** mudanças de prova, calibração (TRI) e perfil dos candidatos entre edições; 2022 e 2023 são edições pós-pandemia, e edições anteriores (2020–2021) tiveram comparecimento atípico.
- **Monitoramento:** nenhum — o modelo não está implantado. TODO(owner): se for servido, monitorar drift das respostas do questionário e o RMSE com notas reais de cada nova edição.
- **Critério de retreino / descontinuação:** reexecutar `make pipeline` a cada nova edição do INEP e revalidar; descontinuar o uso se o RMSE da nova edição sair do IC 95% reportado aqui sem explicação.
