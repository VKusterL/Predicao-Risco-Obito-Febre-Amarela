# Modelos Finais

Esta pasta contém os três modelos CatBoost finais treinados, calibrados e empacotados conforme o pipeline descrito no artigo. Cada arquivo `.joblib` contém o pipeline completo (pré-processador + modelo + calibrador + limiar de decisão otimizado).

## Modelos disponíveis

| Arquivo                                              | Estágio | Atributos | PR-AUC (holdout) | ROC-AUC (holdout) | Limiar (t*) |
| :--------------------------------------------------- | :-----: | :-------: | :--------------: | :---------------: | :---------: |
| `final_model_wrapped_catboost_model1.joblib`         |  **M1** |     7     |      0,226       |       0,680       |    0,088    |
| `final_model_wrapped_catboost_model2.joblib`         |  **M2** |    11     |      0,321       |       0,764       |    0,090    |
| `final_model_wrapped_catboost_model3.joblib`         |  **M3** |    18     |      0,425       |       0,813       |    0,099    |

> [!NOTE]
> O modelo M3 disponibilizado corresponde à variante sem o atributo `HOSPITALIZACAO` (que era redundante em relação a `INTERNACAO`, mantido nos preditores). Por isso, o conjunto efetivo de preditores em M3 é de 18 atributos. Veja [`docs/dicionario_dados.md`](../docs/dicionario_dados.md) para detalhes.

## Estrutura do *wrapper* (`FinalModelWithThreshold`)

Cada arquivo `.joblib` armazena uma instância de uma classe leve que encapsula:

- `model` — o estimador subjacente, podendo ser:
  - `CalibratedClassifierCV` (M1, M2) com pipeline interno (`prep` + `model`);
  - `ProbaCalibratedWrapper` (M3) que combina o pipeline base com um `ProbCalibrator`.
- `threshold` — o limiar de decisão t* ajustado por OOF sob restrição `recall ≥ 0,8`.
- `model_name` — nome do algoritmo (sempre `"catboost"`).
- `metadata` — dicionário com hiperparâmetros, *hash* dos dados e configuração da calibração.

## Carregando e usando os modelos

> [!IMPORTANT]
> Os modelos foram salvos com classes auxiliares definidas no Notebook 2. Para carregá-los em scripts standalone, use o utilitário [`wrappers.py`](../wrappers.py) na raiz do repositório, que recria todas as classes necessárias e suporta as duas convenções de atributos diferentes que apareceram entre versões do código.

```python
import sys, joblib
sys.path.insert(0, "..")             # caminho até wrappers.py

import wrappers, __main__
for name in ["FinalModelWithThreshold", "ProbaCalibratedWrapper", "ProbCalibrator",
             "SafeCatBoostClassifier", "NominalPreprocessor", "get_proba"]:
    setattr(__main__, name, getattr(wrappers, name))

# 1) Carregar
modelo = joblib.load("final_model_wrapped_catboost_model3.joblib")

# 2) Inspecionar
print("Limiar:", modelo.threshold)
print("Algoritmo:", modelo.model_name)
print("Metadata keys:", list(modelo.metadata.keys()))

# 3) Inferir
# X_novo deve conter as mesmas colunas usadas no treinamento
# (ver docs/dicionario_dados.md). Variáveis categóricas devem
# manter as mesmas categorias originais.
proba = modelo.predict_proba(X_novo)[:, 1]   # P(óbito)
pred  = (proba >= modelo.threshold).astype(int)
```

## Reconstruindo o ambiente

> [!CAUTION]
> Para evitar avisos de versão ao carregar os modelos, recomenda-se utilizar as mesmas versões das bibliotecas usadas no treinamento. As figuras deste repositório foram regeradas em Python 3.12 com saída idêntica à do artigo (Tabela 4), o que confirma robustez do pipeline.

## Observação importante sobre uso clínico

> [!WARNING]
> Os modelos disponibilizados têm finalidade exclusivamente científica e exploratória. Eles não constituem dispositivo médico nem substituem julgamento clínico, e não foram validados externamente nem prospectivamente em ambientes assistenciais reais. Qualquer aplicação prática requer validação adicional, aprovação ética/regulatória e supervisão de profissionais qualificados.
