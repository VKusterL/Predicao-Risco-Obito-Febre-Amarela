# Dicionário de Dados — Material Suplementar

Este documento descreve os atributos do conjunto de dados utilizado no artigo *"Predição de Risco de Óbito por Febre Amarela em Diferentes Estágios do Acompanhamento Clínico usando Aprendizado de Máquina"* (SBCAS 2026).

Os dados são oriundos do **Sistema de Informação de Agravos de Notificação (SINAN)** e foram disponibilizados pelo Ministério da Saúde do Brasil mediante solicitação formal via e-SIC (Sistema Eletrônico do Serviço de Informação ao Cidadão). A versão consolidada após pré-processamento contém **17.147 instâncias** e até **18 atributos preditores**, além da variável-alvo binária `OBITO`.

> [!NOTE]
> O dicionário oficial do SINAN está disponível em:
> https://portalsinan.saude.gov.br/images/documentos/Agravos/via/DIC_DADOS_NET_Violencias_v5.pdf

## Composição dos atributos por base

A modelagem foi estruturada em **três bases**, refletindo a disponibilidade temporalmente coerente das informações no fluxo assistencial:

### Base 1 — Notificação (M1) · 7 atributos
`REG_NOTIF`, `IDADE`, `SEXO`, `VACINA_FA`, `AUTOCTONE`, `PERI_SAZONAL`, `DIAS_ATE_NOTIF`

### Base 2 — Pós-notificação (M2) · 11 atributos
Base 1 + `SINT_HEMORRAGICO`, `DISTUR_RENAL`, `SINAL_FAGET`, `DOR_ABDOMINAL`

### Base 3 — Evolução clínica (M3) · 18 atributos (variante usada no paper)
Base 2 + `INTERNACAO`, `RESULT_IGM`, `RESULT_PCR`, `HISTOPATOLOGIA`, `IMUNOHISTOQUIMICA`, `R_ISOVIRAL`, `CRIT_CONF`

> [!IMPORTANT]
> A base original `baseModelo3-FatoresAssociadosAoObito.xlsx` contém também a coluna `HOSPITALIZACAO`, removida na execução do Notebook 2 por ser redundante em relação a `INTERNACAO`. O modelo M3 publicado utiliza, portanto, **18 atributos** (e não 19).

## Descrição completa dos atributos

### Atributos sociodemográficos e geográficos

| Atributo       | Tipo       | Descrição                                                                 | Codificação                                                                                  |
| :------------- | :--------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------- |
| `REG_NOTIF`    | Categórico | Macrorregião do estado de notificação do caso.                            | `1` = Norte, `2` = Nordeste, `3` = Sudeste, `4` = Sul, `5` = Centro-Oeste                    |
| `IDADE`        | Numérico   | Idade do paciente em anos completos.                                      | Inteiro ≥ 0                                                                                  |
| `SEXO`         | Categórico | Sexo do paciente.                                                         | `1` = Masculino, `2` = Feminino, `9` = Ignorado                                              |

### Estado vacinal

| Atributo     | Tipo       | Descrição                                                                                        | Codificação                                              |
| :----------- | :--------- | :----------------------------------------------------------------------------------------------- | :------------------------------------------------------- |
| `VACINA_FA`  | Categórico | Estado vacinal para febre amarela no momento da notificação.                                     | `1` = Vacinado · `2` = Não vacinado · `9` = Ignorado |

### Atributos clínicos e indicadores de gravidade

| Atributo            | Tipo       | Descrição                                                                                  | Codificação                          |
| :------------------ | :--------- | :----------------------------------------------------------------------------------------- | :----------------------------------- |
| `SINT_HEMORRAGICO`  | Categórico | Presença de sintomas hemorrágicos.                                                         | `1` = Sim · `2` = Não · `9` = Ign.   |
| `DISTUR_RENAL`      | Categórico | Presença de distúrbio renal (oligúria/anúria, elevação de creatinina ou ureia).            | `1` = Sim · `2` = Não · `9` = Ign.   |
| `SINAL_FAGET`       | Categórico | Sinal de Faget (bradicardia relativa associada à febre).                                   | `1` = Sim · `2` = Não · `9` = Ign.   |
| `DOR_ABDOMINAL`     | Categórico | Presença de dor abdominal.                                                                 | `1` = Sim · `2` = Não · `9` = Ign.   |
| `INTERNACAO`        | Categórico | Indicador de internação hospitalar do paciente.                                            | `1` = Sim · `2` = Não · `9` = Ign.   |

### Atributos de diagnóstico laboratorial

| Atributo            | Tipo       | Descrição                                                                                            | Codificação                                                  |
| :------------------ | :--------- | :--------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- |
| `RESULT_IGM`        | Categórico | Resultado da sorologia IgM para febre amarela.                                                       | `1` = Reagente · `2` = Não reagente · `3` = Inconclusivo · `4` = Não realizado · `9` = Ign. |
| `RESULT_PCR`        | Categórico | Resultado da PCR (RT-PCR).                                                                           | `1` = Detectável · `2` = Não detectável · `3` = Inconcl. · `4` = Não realizado · `9` = Ign. |
| `HISTOPATOLOGIA`    | Categórico | Achados histopatológicos compatíveis com febre amarela.                                              | `1` = Compatível · `2` = Não compatível · `3` = Inconcl. · `4` = Não realizado · `9` = Ign. |
| `IMUNOHISTOQUIMICA` | Categórico | Detecção de antígeno viral por imuno-histoquímica em tecido.                                          | `1` = Reagente · `2` = Não reagente · `3` = Inconcl. · `4` = Não realizado · `9` = Ign. |
| `R_ISOVIRAL`        | Categórico | Resultado do isolamento viral em cultura.                                                             | `1` = Positivo · `2` = Negativo · `3` = Inconcl. · `4` = Não realizado · `9` = Ign. |

### Classificação epidemiológica do caso

| Atributo     | Tipo       | Descrição                                                                                              | Codificação                                                                |
| :----------- | :--------- | :----------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- |
| `CRIT_CONF`  | Categórico | Critério adotado para confirmação do caso.                                                             | `1` = Laboratorial · `2` = Clínico-epidemiológico · `3` = Em investigação  |
| `AUTOCTONE`  | Categórico | Indica se o caso é autóctone do município de notificação.                                              | `1` = Sim · `2` = Não · `9` = Ign.                                         |

### Atributos temporais derivados

| Atributo          | Tipo       | Descrição                                                                                  | Codificação / Faixa                                                |
| :---------------- | :--------- | :----------------------------------------------------------------------------------------- | :----------------------------------------------------------------- |
| `DIAS_ATE_NOTIF`  | Numérico   | Intervalo (em dias) entre o início dos sintomas e a notificação do caso.                   | Inteiro ≥ 0                                                        |
| `PERI_SAZONAL`    | Categórico | Período sazonal de transmissão da febre amarela.                                           | `1` = Sazonal (dez–abr) · `2` = Não sazonal (mai–ago)               |

### Variável-alvo

| Atributo  | Tipo       | Descrição                                                          | Codificação                |
| :-------- | :--------- | :----------------------------------------------------------------- | :------------------------- |
| `OBITO`   | Binário    | Desfecho do paciente: ocorrência ou não de óbito por febre amarela. | `1` = Óbito · `0` = Sobreviveu / sem óbito registrado |

> [!IMPORTANT]
> Códigos numéricos do SINAN utilizados para representar respostas desconhecidas, inconclusivas ou não coletadas (tipicamente `9`, e variantes como `99`, `999`) foram **consolidados como valores ausentes** durante o pré-processamento, conforme boas práticas de análise de dados em saúde [Little & Rubin, 2019].
