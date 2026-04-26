# Resultados Quantitativos

## Conteúdo

| Arquivo                          | Conteúdo                                                                          |
| :------------------------------- | :-------------------------------------------------------------------------------- |
| `ranking_modelos_modelo1.csv`    | Ranking dos 5 algoritmos na *nested CV* — Base 1 (M1)                              |
| `ranking_modelos_modelo2.csv`    | Ranking dos 5 algoritmos na *nested CV* — Base 2 (M2)                              |
| `ranking_modelos_modelo3.csv`    | Ranking dos 5 algoritmos na *nested CV* — Base 3 (M3)                              |
| `holdout_metrics_full.csv`       | Métricas finais no holdout para os 3 modelos CatBoost selecionados                 |

## Algoritmo vencedor por base

| Base | Atributos | Algoritmo vencedor (PR-AUC nCV) |
| :--: | :-------: | :-----------------------------: |
| M1   |     7     | **CatBoost** (0,248 ± 0,028)     |
| M2   |    11     | **CatBoost** (0,333 ± 0,032)     |
| M3   |    18     | **CatBoost** (0,447 ± 0,042)     |

## Colunas dos rankings (`ranking_modelos_modeloX.csv`)

| Coluna                        | Descrição                                                                                          |
| :---------------------------- | :------------------------------------------------------------------------------------------------- |
| `rank`                        | Posição final no ranking (1 = melhor desempenho médio em PR-AUC)                                   |
| `modelo`                      | Nome do algoritmo                                                                                  |
| `pr_auc_mean / std`           | PR-AUC média e desvio padrão nos 10 *folds* externos                                               |
| `roc_auc_mean / std`          | ROC-AUC média e desvio padrão                                                                      |
| `brier_mean`                  | *Brier score* médio                                                                                |
| `f1@0.5_mean`                 | F1-score médio com limiar fixo em 0,5                                                              |
| `recall@0.5_mean`             | Sensibilidade média com limiar 0,5                                                                 |
| `precision@0.5_mean`          | Precisão média com limiar 0,5                                                                      |
| `bal_acc@0.5_mean`            | Acurácia balanceada média com limiar 0,5                                                           |
| `mcc@0.5_mean`                | MCC médio com limiar 0,5                                                                           |
| `oof_threshold`               | Limiar otimizado nas predições *out-of-fold* (sob restrição `recall ≥ 0,7`)                          |
| `oof_f1/recall/precision/mcc` | Métricas OOF avaliadas no limiar otimizado                                                         |

## Colunas de `holdout_metrics_full.csv`

Métricas computadas pelo script [`generate_figures.py`](../generate_figures.py) ao aplicar cada modelo final ao seu respectivo holdout (mantendo `random_state=42`):

| Coluna       | Descrição                                                  |
| :----------- | :--------------------------------------------------------- |
| `Modelo`     | M1 / M2 / M3                                               |
| `ROC-AUC`    | Área sob a curva ROC                                       |
| `PR-AUC`     | Área sob a curva Precision-Recall (Average Precision)      |
| `Brier`      | *Brier score* (menor é melhor)                             |
| `Sens`       | Sensibilidade (recall)                                     |
| `Prec`       | Precisão                                                    |
| `F1`         | F1-score                                                    |
| `BalAcc`     | Acurácia balanceada                                         |
| `Espec`      | Especificidade                                              |
| `VPN`        | Valor preditivo negativo                                    |
| `MCC`        | Coeficiente de correlação de Matthews                      |
| `Youden`     | Índice de Youden (Sens + Espec − 1)                        |
| `tn/fp/fn/tp`| Contagens da matriz de confusão                            |
| `threshold`  | Limiar de decisão t* utilizado                             |

> Análise consolidada em [`docs/resultados_completos.md`](../docs/resultados_completos.md).
