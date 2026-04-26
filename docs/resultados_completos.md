# Resultados Completos — Material Suplementar

Este documento traz as **tabelas estendidas** que complementam o artigo *"Predição de Risco de Óbito por Febre Amarela em Diferentes Estágios do Acompanhamento Clínico usando Aprendizado de Máquina"* (SBCAS 2026).

## 1. Ranking dos algoritmos na *Nested Cross-Validation*

As tabelas a seguir reportam o desempenho médio (μ) e desvio padrão (σ) dos **10 *folds* externos** da validação cruzada aninhada para cada uma das três bases. A métrica primária de seleção foi **PR-AUC**, com desempate por ROC-AUC e *Brier score*.

> Os valores brutos estão em [`results/ranking_modelos_modelo1.csv`](../results/ranking_modelos_modelo1.csv), [`results/ranking_modelos_modelo2.csv`](../results/ranking_modelos_modelo2.csv) e [`results/ranking_modelos_modelo3.csv`](../results/ranking_modelos_modelo3.csv). Visualização gráfica em [`images/algorithm_comparison.png`](../images/algorithm_comparison.png).

### 1.1 Base 1 (M1 — Notificação)

| Rank | Algoritmo       | PR-AUC (μ ± σ)        | ROC-AUC (μ ± σ)       | Brier (μ)  | F1@0.5 | OOF *t* | OOF F1 |
| :--: | :-------------- | :--------------------: | :--------------------: | :--------: | :----: | :-----: | :----: |
|  1   | **CatBoost**     | **0,248 ± 0,028**     | **0,688 ± 0,026**     | **0,215**  | 0,297  |  0,459  | 0,286  |
|  2   | Random Forest    | 0,247 ± 0,030         | 0,685 ± 0,025         | 0,212      | 0,295  |  0,461  | 0,288  |
|  3   | XGBoost          | 0,241 ± 0,030         | 0,684 ± 0,030         | 0,220      | 0,290  |  0,467  | 0,286  |
|  4   | AdaBoost         | 0,227 ± 0,028         | 0,673 ± 0,031         | 0,239      | 0,287  |  0,492  | 0,285  |
|  5   | Decision Tree    | 0,204 ± 0,026         | 0,646 ± 0,029         | 0,226      | 0,273  |  0,431  | 0,263  |

### 1.2 Base 2 (M2 — Pós-notificação)

| Rank | Algoritmo       | PR-AUC (μ ± σ)        | ROC-AUC (μ ± σ)       | Brier (μ)  | F1@0.5 | OOF *t* | OOF F1 |
| :--: | :-------------- | :--------------------: | :--------------------: | :--------: | :----: | :-----: | :----: |
|  1   | **CatBoost**     | **0,333 ± 0,032**     | **0,767 ± 0,022**     | **0,189**  | 0,361  |  0,485  | 0,354  |
|  2   | XGBoost          | 0,330 ± 0,036         | 0,762 ± 0,024         | 0,198      | 0,346  |  0,489  | 0,345  |
|  3   | Random Forest    | 0,329 ± 0,038         | 0,762 ± 0,024         | 0,185      | 0,357  |  0,475  | 0,345  |
|  4   | AdaBoost         | 0,314 ± 0,037         | 0,748 ± 0,029         | 0,227      | 0,340  |  0,494  | 0,337  |
|  5   | Decision Tree    | 0,289 ± 0,031         | 0,730 ± 0,026         | 0,201      | 0,336  |  0,446  | 0,310  |

### 1.3 Base 3 (M3 — Evolução clínica)

| Rank | Algoritmo       | PR-AUC (μ ± σ)        | ROC-AUC (μ ± σ)       | Brier (μ)  | F1@0.5 | OOF *t* | OOF F1 |
| :--: | :-------------- | :--------------------: | :--------------------: | :--------: | :----: | :-----: | :----: |
|  1   | **CatBoost**     | **0,447 ± 0,042**     | **0,827 ± 0,017**     | **0,162**  | 0,418  |  0,511  | 0,423  |
|  2   | XGBoost          | 0,446 ± 0,046         | 0,824 ± 0,018         | 0,160      | 0,419  |  0,505  | 0,422  |
|  3   | Random Forest    | 0,419 ± 0,044         | 0,817 ± 0,020         | 0,167      | 0,413  |  0,493  | 0,409  |
|  4   | AdaBoost         | 0,412 ± 0,044         | 0,807 ± 0,021         | 0,224      | 0,394  |  0,501  | 0,397  |
|  5   | Decision Tree    | 0,332 ± 0,038         | 0,759 ± 0,027         | 0,188      | 0,378  |  0,465  | 0,343  |

> [!TIP]
> Em todas as três bases, **CatBoost** apresentou o melhor desempenho médio em PR-AUC, com **estabilidade ao longo dos *folds* externos** (desvios padrão entre 0,028 e 0,042). Esse resultado consistente reduz a probabilidade de viés otimista associado à seleção do algoritmo [Cawley & Talbot, 2010] e justifica a escolha do CatBoost para o refinamento profundo na Etapa 2.

## 2. Desempenho do modelo final (CatBoost) no *holdout*

Modelos finais treinados após o refinamento com Optuna (300 *trials*, *sampler* TPE, *pruning*) e ajuste de limiar OOF sob restrição `recall ≥ 0,8`.

### 2.1 Métricas pontuais (idênticas à Tabela 4 do paper)

| Modelo | ROC-AUC | PR-AUC | Brier | *t* * | Sens. | Prec. | F1   | AcB  | Es   | VPN  | MCC   | IY    |
| :----: | :-----: | :----: | :---: | :---: | :---: | :---: | :---: | :--: | :--: | :--: | :---: | :---: |
| **M1** | 0,680  | 0,226  | 0,101 | 0,088 | 0,761 | 0,162 | 0,267 | 0,609 | 0,456 | 0,933 | 0,143 | 0,218 |
| **M2** | 0,764  | 0,321  | 0,094 | 0,090 | 0,817 | 0,202 | 0,324 | 0,687 | 0,556 | 0,957 | 0,243 | 0,373 |
| **M3** | 0,813  | 0,425  | 0,087 | 0,099 | 0,841 | 0,228 | 0,359 | 0,725 | 0,609 | 0,965 | 0,295 | 0,450 |

> Visualização: [`images/metrics_heatmap.png`](../images/metrics_heatmap.png), [`images/metrics_radar.png`](../images/metrics_radar.png) e [`images/performance_evolution.png`](../images/performance_evolution.png).

### 2.2 Comparação operacional (matrizes de confusão normalizadas por classe real)

| Modelo | TN%       | FP%       | FN%       | TP%       | Limiar | TN | FP | FN | TP |
| :----: | :-------: | :-------: | :-------: | :-------: | :----: | :--: | :--: | :--: | :--: |
| **M1** | 45,6%     | 54,4%     | 23,9%     | 76,1%     | 0,088  | 1376 | 1639 |  99 | 316 |
| **M2** | 55,6%     | 44,4%     | 18,3%     | 81,7%     | 0,090  | 1677 | 1338 |  76 | 339 |
| **M3** | 60,9%     | 39,1%     | 15,9%     | 84,1%     | 0,099  | 1835 | 1180 |  66 | 349 |

> Visualização: [`images/figure1_combined.png`](../images/figure1_combined.png) e individuais em [`images/M{1,2,3}/confusion_matrix.png`](../images/M1/).

### 2.3 Evolução do ganho preditivo

| Estágio   | ROC-AUC | PR-AUC | Múltiplo do acaso (π = 0,121) |
| :-------- | :-----: | :----: | :----------------------------: |
| **M1**    | 0,680  | 0,226  | **1,9× o acaso**               |
| **M2**    | 0,764  | 0,321  | **2,7× o acaso**                |
| **M3**    | 0,813  | 0,425  | **3,5× o acaso**                |

A redução paralela do *Brier score* (de 0,101 → 0,094 → 0,087) confirma o aprimoramento da **calibração e da qualidade das probabilidades estimadas** ao longo dos estágios [Gneiting & Raftery, 2007; Steyerberg et al., 2010].

## 3. Análise SHAP — Ranking de importância

Atributos ordenados por importância média absoluta (|SHAP|) calculada sobre o conjunto de *holdout*.

### 3.1 Modelo M1 (notificação)

| Posição | Atributo            | Categoria                         |
| :-----: | :------------------ | :-------------------------------- |
|    1    | `IDADE`              | Sociodemográfico                  |
|    2    | `PERI_SAZONAL`       | Temporal                          |
|    3    | `AUTOCTONE`          | Classificação epidemiológica      |
|    4    | `VACINA_FA`          | Estado vacinal                    |
|    5    | `DIAS_ATE_NOTIF`     | Temporal                          |
|    6    | `REG_NOTIF`          | Geográfico                        |
|    7    | `SEXO`               | Sociodemográfico                  |

### 3.2 Modelo M2 (pós-notificação)

| Posição | Atributo               | Categoria                          |
| :-----: | :--------------------- | :--------------------------------- |
|    1    | `SINT_HEMORRAGICO`      | Clínico                            |
|    2    | `IDADE`                 | Sociodemográfico                   |
|    3    | `DISTUR_RENAL`          | Clínico                            |
|    4    | `PERI_SAZONAL`          | Temporal                           |
|    5    | `VACINA_FA`             | Estado vacinal                     |
|    6    | `DIAS_ATE_NOTIF`        | Temporal                           |
|    7    | `AUTOCTONE`             | Classificação epidemiológica       |
|    8    | `REG_NOTIF`             | Geográfico                         |
|    9    | `DOR_ABDOMINAL`         | Clínico                            |
|   10    | `SEXO`                  | Sociodemográfico                   |
|   11    | `SINAL_FAGET`           | Clínico                            |

### 3.3 Modelo M3 (evolução clínica)

| Posição | Atributo               | Categoria                          |
| :-----: | :--------------------- | :--------------------------------- |
|    1    | `SINT_HEMORRAGICO`      | Clínico                            |
|    2    | `INTERNACAO`            | Clínico                            |
|    3    | `DISTUR_RENAL`          | Clínico                            |
|    4    | `IDADE`                 | Sociodemográfico                   |
|    5    | `RESULT_PCR`            | Laboratorial                       |
|    6    | `PERI_SAZONAL`          | Temporal                           |
|    7    | `AUTOCTONE`             | Classificação epidemiológica       |
|    8    | `DOR_ABDOMINAL`         | Clínico                            |
|    9    | `CRIT_CONF`             | Classificação epidemiológica       |
|   10    | `VACINA_FA`             | Estado vacinal                     |
|   11    | `R_ISOVIRAL`            | Laboratorial                       |
|   12    | `REG_NOTIF`             | Geográfico                         |
|   13    | `RESULT_IGM`            | Laboratorial                       |
|   14    | `IMUNOHISTOQUIMICA`     | Laboratorial                       |
|   15    | `DIAS_ATE_NOTIF`        | Temporal                           |
|   16    | `SINAL_FAGET`           | Clínico                            |
|   17    | `SEXO`                  | Sociodemográfico                   |
|   18    | `HISTOPATOLOGIA`        | Laboratorial                       |

> [!IMPORTANT]
> A análise mostra uma **transição de protagonismo entre as categorias de atributos** ao longo dos estágios: do domínio de variáveis demográficas/epidemiológicas em M1, para a centralidade dos sinais clínicos em M2, culminando com a relevância dos exames laboratoriais e indicadores de gravidade em M3 — padrão alinhado com a fisiopatologia e o curso natural da doença.

> Visualização individual: [`images/M1/shap_summary.png`](../images/M1/shap_summary.png), [`images/M2/shap_summary.png`](../images/M2/shap_summary.png), [`images/M3/shap_summary.png`](../images/M3/shap_summary.png). Painel combinado: [`images/figure2_shap_combined.png`](../images/figure2_shap_combined.png).

## 4. Configurações de busca de hiperparâmetros

### 4.1 Etapa 1 — Seleção de modelo (BayesSearchCV, 90 iterações por algoritmo)

| Algoritmo       | Hiperparâmetros e espaços                                                                                                                                                                                                                                                                                                                  |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Decision Tree   | `max_depth ∈ [1, 30]` · `min_samples_split ∈ [2, 50]` · `min_samples_leaf ∈ [1, 20]` · `criterion ∈ {gini, entropy}`                                                                                                                                                                                                                       |
| Random Forest   | `n_estimators ∈ [200, 1200]` · `max_depth ∈ [2, 40]` · `min_samples_split ∈ [2, 50]` · `min_samples_leaf ∈ [1, 20]` · `max_features ∈ [0,2; 1,0]` · `bootstrap ∈ {True, False}`                                                                                                                                                              |
| AdaBoost        | `n_estimators ∈ [50, 600]` · `learning_rate ∈ [10⁻³; 2,0]` (log-uniforme)                                                                                                                                                                                                                                                                   |
| XGBoost         | `n_estimators ∈ [200, 1500]` · `max_depth ∈ [2, 10]` · `learning_rate ∈ [10⁻³; 0,3]` (log-unif.) · `subsample ∈ [0,5; 1,0]` · `colsample_bytree ∈ [0,5; 1,0]` · `min_child_weight ∈ [0,5; 20,0]` (log-unif.) · `gamma ∈ [0,0; 5,0]` · `reg_lambda ∈ [10⁻³; 100]` (log-unif.)                                                                  |
| CatBoost        | `iterations ∈ [200, 2000]` · `depth ∈ [4, 10]` · `learning_rate ∈ [10⁻³; 0,3]` (log-unif.) · `l2_leaf_reg ∈ [10⁻³; 100]` (log-unif.)                                                                                                                                                                                                          |

### 4.2 Etapa 2 — Refino profundo (Optuna, 300 *trials*, sampler TPE com *pruning*)

A busca profunda expandiu os espaços para o algoritmo vencedor (CatBoost), mantendo as faixas listadas na Tabela 3 do artigo. Os melhores hiperparâmetros encontrados em cada base são serializados em `final_report.json` no `OUTPUT_DIR` correspondente, gerados pelo Notebook 2.

## Referências

- Cawley, G. C. & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*.
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477).
- Lundberg, S. M. & Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS One*, 10(3).
- Steyerberg, E. et al. (2010). Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*, 21(1).
