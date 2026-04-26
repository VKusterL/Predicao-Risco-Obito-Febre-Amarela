# Bases de Dados

Esta pasta contém as três bases pré-processadas utilizadas neste estudo, uma para cada estágio do acompanhamento clínico modelado.

| Arquivo                                          | Modelo associado | Atributos | Instâncias | Casos com óbito |
| :----------------------------------------------- | :--------------: | :-------: | :--------: | :-------------: |
| `baseModelo1-VigilanciaNaNotificacao.xlsx`       |        M1        |     7     |   17.147   |  2.076 (12,1%)  |
| `baseModelo2-RiscoClinicoInicial.xlsx`           |        M2        |    11     |   17.147   |  2.076 (12,1%)  |
| `baseModelo3-FatoresAssociadosAoObito.xlsx`      |        M3        |    19*    |   17.147   |  2.076 (12,1%)  |

> [!NOTE]
> A base do modelo M3 contém 19 colunas preditoras (incluindo `HOSPITALIZACAO`), mas o modelo final treinado utiliza 18, removendo `HOSPITALIZACAO` por redundância em relação a `INTERNACAO`. Essa remoção é feita programaticamente pelo Notebook 2 e pelo script [`generate_figures.py`](../generate_figures.py).

## Origem dos dados

As bases foram derivadas de dados secundários do **Sistema de Informação de Agravos de Notificação (SINAN)** do Ministério da Saúde do Brasil, obtidos mediante solicitação formal pelo **Sistema Eletrônico do Serviço de Informação ao Cidadão (e-SIC)**. A versão original cobre o período 2000–2018, com 17.905 registros e 27 atributos brutos.

## Pré-processamento aplicado

Os arquivos disponibilizados nesta pasta já refletem as seguintes etapas de pré-processamento:

1. Identificação e tratamento de duplicatas.
2. Consolidação dos diferentes códigos do SINAN para representar valores ausentes (tipicamente `9`, `99`, `999`) em uma representação unificada.
3. Remoção de atributos redundantes ou pouco informativos (e.g., `UF_NOTIF` em relação a `REG_NOTIF`, `FAIXA_ETARIA` em relação a `IDADE`, `CLASSI_FIN` em relação a `CRIT_CONF`).
4. Engenharia de variáveis temporais (`DIAS_ATE_NOTIF`, `MES_SINTOMAS`, `PERI_SAZONAL`) a partir dos campos brutos de data.
5. Remoção das instâncias que ainda apresentavam valores ausentes após os passos anteriores, resultando em 17.147 registros (de 17.905 originais).

Esses passos estão alinhados com boas práticas de análise de dados em saúde [Little & Rubin, 2019] e com o dicionário oficial do SINAN.

## Codificação dos atributos

A descrição completa de cada coluna — nome, tipo, descrição clínica, categorias e codificação numérica — está em [`docs/dicionario_dados.md`](../docs/dicionario_dados.md). Variáveis como `SEXO`, `VACINA_FA`, `RESULT_IGM` etc. são representadas por inteiros, conforme convenção do SINAN.

## Considerações éticas

Os dados não contêm informações nominais, identificadores diretos (CPF, nome, endereço) ou outros dados pessoais sensíveis que permitam reidentificação direta dos pacientes. As variáveis disponíveis são agregadas por região, faixa etária ou ano e foram analisadas exclusivamente para fins de pesquisa científica.

> [!IMPORTANT]
> Ao utilizar estas bases em pesquisas derivadas, recomenda-se citar o artigo original e o Ministério da Saúde do Brasil como fonte primária dos dados.
