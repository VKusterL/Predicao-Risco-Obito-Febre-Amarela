# Reprodutibilidade — Material Suplementar

Este documento detalha as configurações utilizadas para garantir a reprodutibilidade dos experimentos descritos no artigo *"Predição de Risco de Óbito por Febre Amarela em Diferentes Estágios do Acompanhamento Clínico usando Aprendizado de Máquina"* (SBCAS 2026).

> [!TIP]
> Este repositório foi **validado** por meio de regeneração completa de todas as figuras (script [`generate_figures.py`](../generate_figures.py)) em ambiente Python 3.12 — os números obtidos coincidiram exatamente com os reportados na Tabela 4 do paper, o que confirma a robustez do pipeline.

## 1. Ambiente Computacional

| Componente            | Especificação (paper)                                      | Especificação (validação)                |
| :-------------------- | :--------------------------------------------------------- | :----------------------------------------- |
| Sistema operacional   | Windows 11                                                 | Linux x86_64                                |
| Python                | 3.13                                                       | 3.12                                        |
| Hardware              | Intel Core i7-7700K · 16 GB RAM · NVIDIA GTX 1070 (8 GB)   | CPU                                         |
| Execução              | Predominantemente em CPU                                   | CPU                                         |
| IDE                   | Jupyter Notebook                                           | Jupyter Notebook                            |

> [!NOTE]
> A reprodução foi numericamente idêntica em uma versão diferente de Python (3.12 vs 3.13), o que demonstra portabilidade do pipeline. A GPU não é necessária — algoritmos baseados em árvores são executados eficientemente em CPU.

## 2. Sementes Aleatórias (*Random Seeds*)

Todas as fontes de aleatoriedade foram fixadas para garantir reprodutibilidade:

| Componente                                   | Valor                                  |
| :------------------------------------------- | :------------------------------------- |
| `RANDOM_STATE` global                         | `42`                                   |
| `train_test_split` (separação do *holdout*)   | `random_state=42`                      |
| `RepeatedStratifiedKFold` (lacço externo)     | `random_state=42`                      |
| `StratifiedKFold` (lacço interno)             | `random_state=42`                      |
| `BayesSearchCV`                               | `random_state=42 + 777 = 819`           |
| Optuna *sampler* TPE                          | `seed=42`                              |
| OOF para limiar (`RS_OOF`)                    | `42`                                   |
| Calibração (`RS_CALIB`)                       | `42`                                   |
| *Bootstrap* CI                                | `random_state=42`                      |

## 3. Validação cruzada e protocolo de avaliação

### Etapa 1 — Seleção de modelo

| Parâmetro                       | Valor                                       |
| :------------------------------ | :------------------------------------------ |
| Tamanho do *holdout*            | 20% (estratificado)                         |
| Lacço externo                   | 10 *folds* estratificados, 1 repetição      |
| Lacço interno                   | 10 *folds* estratificados                   |
| Iterações `BayesSearchCV`       | 90 por algoritmo × base                     |
| Métrica de otimização           | `average_precision` (PR-AUC)                |
| Métrica para limiar             | `precision` (sob restrição `recall ≥ 0,7`)   |

### Etapa 2 — Refino profundo

| Parâmetro                              | Valor                                        |
| :------------------------------------- | :------------------------------------------- |
| Iterações Optuna                       | 300 *trials*                                 |
| *Sampler*                              | `TPESampler` (Tree-structured Parzen Estimator) |
| *Pruner*                               | `MedianPruner`                               |
| *Folds* CV interna                     | 10 estratificados                            |
| *Early stopping* (XGBoost / CatBoost)  | Aplicado em cada *fold*                      |
| OOF para limiar                        | 10 *folds* × 3 repetições                    |
| Métrica para limiar                    | `precision` (sob restrição `recall ≥ 0,8`)   |
| *Bootstrap* CI                         | 2.000 reamostras estratificadas               |
| Nível de confiança                     | 95%                                          |

## 4. *Hash* da base de dados

Para garantir que diferentes execuções utilizem **exatamente a mesma versão dos dados**, o Notebook 2 calcula o SHA-256 do arquivo de entrada e salva o valor em `final_report.json`:

```python
import hashlib

def compute_file_hash(filepath, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

> [!IMPORTANT]
> A base utilizada nos experimentos do artigo possui um *hash* específico, registrado nos relatórios de cada modelo final. Ao replicar o estudo, **verifique se o *hash* da sua base coincide** — divergências indicam que pré-processamento, fontes ou versões dos dados são diferentes.

## 5. Persistência do *holdout*

Os índices de treino/teste para cada base são persistidos em `holdout_split_indices.json`, permitindo replicar **exatamente o mesmo conjunto de avaliação** em execuções futuras:

```json
{
  "train_index": [...],
  "test_index": [...],
  "test_size": 0.20,
  "random_state": 42
}
```

Para o *holdout* utilizado no paper:
- **Treino:** 13.717 instâncias (80%)
- **Teste:** 3.430 instâncias (20%), com 415 óbitos (12,1%)

## 6. Versões de bibliotecas

As principais dependências e versões utilizadas estão em [`requirements.txt`](../requirements.txt). Após cada execução do Notebook 2, um *snapshot* completo do ambiente é salvo em `pip_freeze.txt` no `OUTPUT_DIR` correspondente.

### Bibliotecas críticas

| Biblioteca           | Versão sugerida   | Função                                                         |
| :------------------- | :---------------- | :------------------------------------------------------------- |
| `python`             | 3.12 ou 3.13      | Interpretador (validado em ambas)                              |
| `numpy`              | ≥ 1.26             | Computação numérica                                            |
| `pandas`             | ≥ 2.2              | Manipulação de dados tabulares                                 |
| `scikit-learn`       | ≥ 1.6              | Pipeline, métricas, validação cruzada, calibração              |
| `scikit-optimize`    | ≥ 0.10             | `BayesSearchCV` para otimização bayesiana (Etapa 1)            |
| `optuna`             | ≥ 3.6              | Otimização bayesiana com TPE e *pruning* (Etapa 2)             |
| `catboost`           | ≥ 1.2              | CatBoost (algoritmo vencedor)                                  |
| `xgboost`            | ≥ 2.0              | XGBoost                                                        |
| `joblib`             | ≥ 1.4              | Serialização dos modelos finais                                |
| `shap`               | ≥ 0.45             | Análise de explicabilidade                                     |
| `matplotlib`         | ≥ 3.8              | Geração de figuras                                             |
| `openpyxl`           | ≥ 3.1              | Leitura dos arquivos `.xlsx` das bases                         |

## 7. Política de **vazamento de dados** (*data leakage*)

O protocolo foi desenhado para **eliminar vazamento prospectivo**:

1. **Coerência temporal:** cada base contém somente atributos disponíveis no respectivo momento do acompanhamento. Atributos que só existiriam após o desfecho (e.g., `DT_OBITO`, `DT_ENCERRA`) **nunca** integram o conjunto de preditores.
2. **Isolamento do *holdout*:** o *holdout* (20%) é congelado no início e não participa de nenhuma decisão de modelagem (seleção de algoritmo, otimização de hiperparâmetros, calibração, ajuste de limiar).
3. **Calibração com *cross-fitting*:** quando aplicada, a calibração utiliza `CalibratedClassifierCV` com estratégia `cv` (*cross-fitting*) para evitar que probabilidades de treino sejam usadas para ajustar o calibrador.
4. **Imputação dentro do pipeline:** imputação pela mediana/moda é parte integrante do `Pipeline` do *scikit-learn*, garantindo que estatísticas sejam calculadas apenas sobre os *folds* de treino.
5. **OOF para limiar:** o limiar de decisão é estimado com predições *out-of-fold*, evitando ajuste sobre as mesmas instâncias usadas para treinar o modelo.

## 8. Como replicar passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/VKusterL/Predicao-Risco-Obito-Febre-Amarela.git
cd Predicao-Risco-Obito-Febre-Amarela

# 2. Criar ambiente virtual e instalar dependências
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Solicitar e organizar os dados
# As bases pré-processadas já estão disponíveis em bases/ neste repositório.
# Caso queira partir dos dados brutos do SINAN, solicite via e-SIC e siga o
# pré-processamento descrito no artigo (ver docs/bases/README.md e
# docs/dicionario_dados.md).

# 4. Executar o Notebook 1 para cada base (ajuste BASE_FILE)
jupyter lab notebooks/1-SelecaoModelo.ipynb

# 5. Executar o Notebook 2 para cada base
jupyter lab notebooks/2-BuscaProfundaHiperparametros.ipynb

# 6. (Opcional) Regenerar todas as figuras a partir dos modelos salvos
python generate_figures.py
```

## 9. Verificação dos modelos disponibilizados

Os modelos em [`models/`](../models/) podem ser carregados diretamente para reproduzir as predições no *holdout*. Use o utilitário [`wrappers.py`](../wrappers.py) para lidar com as classes auxiliares definidas durante o treinamento:

```python
import sys, joblib
sys.path.insert(0, ".")  # caminho até wrappers.py

import wrappers, __main__
for name in ["FinalModelWithThreshold", "ProbaCalibratedWrapper", "ProbCalibrator",
             "SafeCatBoostClassifier", "NominalPreprocessor", "get_proba"]:
    setattr(__main__, name, getattr(wrappers, name))

modelo = joblib.load("models/final_model_wrapped_catboost_model3.joblib")
print("Limiar de decisão (t*):", modelo.threshold)

# Aplicar a um conjunto X_test com mesmas colunas e codificações
proba = modelo.predict_proba(X_test)[:, 1]
pred  = (proba >= modelo.threshold).astype(int)
```

> [!CAUTION]
> Os modelos M1 e M2 utilizam internamente `CalibratedClassifierCV`, enquanto M3 usa um `ProbaCalibratedWrapper` customizado. Os atributos do `NominalPreprocessor` também variam entre `_num_imputer/_cat_imputer` (M1, M2) e `_num_imp/_cat_imp` (M3) — o `wrappers.py` neste repositório **suporta ambas as convenções** automaticamente, garantindo desserialização confiável.
