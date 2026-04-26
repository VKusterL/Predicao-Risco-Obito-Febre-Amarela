# Figuras

Esta pasta contém todas as figuras geradas a partir dos modelos finais aplicados ao holdout (20% dos dados), reproduzindo os resultados reportados no artigo. Todas foram produzidas pelo script [`generate_figures.py`](../generate_figures.py) na raiz do repositório.

Para regerar as figuras, basta executar `python generate_figures.py` após instalar as dependências de [`requirements.txt`](../requirements.txt). As bases de dados necessárias já estão disponíveis em [`bases/`](../bases/).

## Figuras consolidadas

| Arquivo                              | Descrição                                                                         |
| :----------------------------------- | :-------------------------------------------------------------------------------- |
| `figure1_combined.png`               | Reproduz a Figura 1 do artigo: matrizes de confusão (esquerda) e histogramas de probabilidade (direita) para M1, M2 e M3 |
| `figure2_shap_combined.png`          | Reproduz a Figura 2 do artigo: gráficos SHAP *beeswarm* lado a lado para M1, M2 e M3 |
| `algorithm_comparison.png`           | Comparação dos 5 algoritmos × 3 bases com barras de erro (PR-AUC e ROC-AUC, *nested CV*) |
| `performance_evolution.png`          | Linhas de evolução do desempenho (ROC-AUC, PR-AUC, Brier) M1 → M2 → M3            |
| `metrics_radar.png`                  | Radar comparativo das métricas operacionais (Sensibilidade, Especificidade, Precisão, VPN, Acurácia balanceada, F1) |
| `metrics_heatmap.png`                | Heatmap consolidado: 11 métricas × 3 modelos no holdout                            |

## Figuras individuais por modelo

Em [`M1/`](./M1/), [`M2/`](./M2/) e [`M3/`](./M3/), cada modelo possui o seguinte conjunto:

| Arquivo                       | Descrição                                                                                       |
| :---------------------------- | :---------------------------------------------------------------------------------------------- |
| `confusion_matrix.png`        | Matriz de confusão normalizada por classe real, com percentuais e contagens absolutas, no limiar t* |
| `probability_histogram.png`   | Distribuição das probabilidades estimadas para a classe óbito, separando acertos e erros        |
| `roc_pr_curves.png`           | Curvas ROC (esquerda) e Precision-Recall (direita) com AUC anotada e linha de referência do acaso |
| `calibration_curve.png`       | Curva de calibração (10 bins por quantis) com linha de calibração perfeita                     |
| `feature_importance.png`      | Importância nativa do CatBoost (PredictionValuesChange) — todos os atributos                    |
| `shap_summary.png`            | *Beeswarm plot* SHAP completo, com cor representando o valor do atributo (low/high)             |
| `shap_importance.png`         | Ranking dos atributos por importância média absoluta SHAP                                        |

## Verificação numérica

Os valores reportados nas figuras coincidem exatamente com a Tabela 4 do artigo:

| Modelo | ROC-AUC | PR-AUC | Brier | Sens.   | Espec.  |
| :----: | :-----: | :----: | :---: | :-----: | :-----: |
| **M1** | 0,680  | 0,226  | 0,101 | 0,761  | 0,456  |
| **M2** | 0,764  | 0,321  | 0,094 | 0,817  | 0,556  |
| **M3** | 0,813  | 0,425  | 0,087 | 0,841  | 0,609  |

> [!NOTE]
> O abstract do artigo menciona PR-AUC = 0,434 para M3, enquanto a Tabela 4 reporta 0,425 (que é o valor obtido pelo modelo). A discrepância parece ser apenas um arredondamento divergente entre o texto do abstract e a tabela.

## Notas sobre estilo e visualização

As variáveis categóricas no SHAP são tratadas como numéricas para fins de visualização (utilizando os códigos inteiros das categorias), o que garante que todos os atributos apareçam coloridos no *beeswarm plot*. Isso preserva a interpretação ordinal sempre que ela é razoável e melhora a leitura do gráfico em comparação com a coloração cinza padrão para variáveis categóricas.

A paleta utilizada é sóbria (gradiente azul-petróleo do M1 mais claro ao M3 mais escuro), com tipografia DejaVu Sans em 10–12 pt e resolução de 200 dpi em formato PNG.
