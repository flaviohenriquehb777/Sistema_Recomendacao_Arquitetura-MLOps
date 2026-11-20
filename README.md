<div align="center">

# Sistema de Recomendações
## NeuMF + MLOps · Recomendação de Produtos

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-0.12-261230.svg)](https://docs.astral.sh/uv/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Enabled-blue.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-green.svg)](https://github.com/features/actions)
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**Recomendador de produtos com Deep Learning (NeuMF), avaliação por métricas de ranking, explicabilidade com SHAP, ambiente MLOps completo (uv + DVC/DagsHub + MLflow) e serviço de inferência empacotado (FastAPI + Docker).**

</div>

---

## Sumário

- [Descrição do Modelo](#descrição-do-modelo)
- [Motivação](#motivação)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Bases de Dados](#bases-de-dados)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Resultados (base real sc_base_01)](#resultados-base-real-sc_base_01)
- [Benchmark em Base Pública Real (sc_base_03)](#benchmark-em-base-pública-real-sc_base_03)
- [Benchmark de Escala (sc_base_04)](#benchmark-de-escala-sc_base_04)
- [Explicabilidade (SHAP)](#explicabilidade-shap)
- [Ambiente MLOps](#ambiente-mlops)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Instalação e Uso](#instalação-e-uso)
- [Testes](#testes)
- [Serviço de Inferência (API)](#serviço-de-inferência-api)
- [Observações Importantes](#observações-importantes)
- [Licença](#licença)
- [Contato](#contato)

---

## Descrição do Modelo

Este modelo implementa um **sistema de recomendação de produtos**, construído sobre a base de vendas *sc_base_01*. O problema é tratado como **filtragem colaborativa neural com objetivo de ranking**: dado um cliente, ordenar o catálogo e devolver os **07 (sete) produtos mais relevantes**, excluindo os itens que o cliente já comprou.

A rede é uma **NeuMF** (Neural Matrix Factorization, He et al., 2017) — combinação de *GMF* (fatoração generalizada) e *MLP* — treinada com **entropia cruzada binária sobre pares cliente–produto** com **amostragem de negativos**, em vez de regressão sobre a coluna `Sales`. Essa é a decisão de modelagem central e está justificada em [Arquitetura do Modelo](#arquitetura-do-modelo).

O modelo final é um **híbrido popularity-first**:

```
score(cliente, produto) = z(log1p(popularidade)) + alpha · z(NeuMF(cliente, produto))
```

onde o *prior* de popularidade (contado no treino) é o sinal-base e o sinal personalizado da NeuMF entra como um ajuste ponderado por `alpha`, escolhido na validação.

No exemplo demonstrado, as recomendações são geradas para uma **cliente de exemplo** da base. O sistema inclui uma função flexível que gera recomendações para **qualquer** consumidor — incluindo **novos clientes** (cold start), que não estavam na base de treino.

## Motivação

Aprimorar a experiência de compra dos clientes por meio de recomendações altamente personalizadas. Ao direcionar cada usuário a produtos que realmente lhe interessam, o sistema busca aumentar a conversão, a satisfação e a lealdade, apoiando as estratégias de *cross-sell* e *upsell* da plataforma.

## Tecnologias Utilizadas

<div align="center">

| Camada | Tecnologias |
|:--|:--|
| Linguagem e ambiente | Python 3.11 · **uv** (fonte única: `pyproject.toml` + `uv.lock`) |
| Dados | pandas · NumPy · PyArrow (Parquet) |
| Modelagem | TensorFlow 2.19 / Keras 3.9 (NeuMF) · scikit-learn · SciPy |
| Explicabilidade | SHAP 0.49 |
| Visualização | Matplotlib · Seaborn · ReportLab (relatórios em PDF) |
| MLOps | **DVC** + **DagsHub** (versionamento) · **MLflow 3** (tracking/registry) |
| Qualidade e CI/CD | pytest · Ruff · Black · GitHub Actions |

</div>

## Bases de Dados

O projeto trabalha com bases identificadas **apenas por nome-código** — `sc_base_01` a `sc_base_05` — para manter a imersão no projeto e isolar a análise de qualquer referência externa. Cada base passa pelas mesmas etapas de **entendimento e tratamento de dados** antes de alimentar o modelo (ver `docs/ROADMAP_RECOMENDACAO_ESCALA.md`).

| Código | Papel no projeto |
|:--|:--|
| `sc_base_01` | **Teste de fumaça de infraestrutura** — transações de varejo, esparsa e de baixa cardinalidade; exercita o pipeline ponta a ponta (ETL, DVC, MLflow, SHAP, cold start). |
| `sc_base_02` | **Reservada** — benchmark de transações em escala; exige aceite de regras na conta do usuário antes do download. |
| `sc_base_03` | **Benchmark real de comportamento** — log de eventos (`view`, `addtocart`, `transaction`). |
| `sc_base_04` | **Benchmark principal de escala** — eventos de interação (`view`, `cart`, `purchase`) de uma loja multicategoria. |
| `sc_base_05` | **Domínio cruzado** — eventos de interação de outra categoria, para validar generalização. |

Na etapa de preparação da `sc_base_01` (`src/scripts/preparar_dados.py`) são aplicados: remoção de valores nulos, correção de tipos (`dtypes`) e *downcast* para reduzir o uso de memória. O resultado é persistido em **Parquet** (`dados/sc_base_01/sc_base_01_tratado.parquet`) para leitura eficiente.

<div align="center">

| Métrica | Valor |
|:--|--:|
| Interações (linhas) | 9.800 |
| Clientes únicos | 793 |
| Produtos únicos | 1.861 |
| Categorias / Subcategorias | 3 / 17 |
| Pares treino / validação / teste | 8.171 / 776 / 776 |
| Amostras de treino (positivos + negativos) | 73.539 |
| Itens por cliente (média / mediana / máx.) | 12,30 / 11 / 34 |
| Densidade da matriz cliente × produto | 0,6607 % |

</div>

> **Nota de contexto:** a densidade de **0,66 %** caracteriza uma base *esparsa de baixa cardinalidade* (793 clientes × 1.861 produtos, ~12 itens por cliente). Esse cenário condiciona as conclusões de modelagem apresentadas em [Resultados (base real sc_base_01)](#resultados-base-real-sc_base_01).

## Arquitetura do Modelo

A NeuMF recebe **quatro entradas** — cliente, produto, categoria e subcategoria — e produz um *score* de interação em `[0, 1]`.

<div align="center">

| Componente | Descrição |
|:--|:--|
| Entradas | `customer`, `product`, `category`, `subcategory` (IDs inteiros) |
| Embeddings | Fatores latentes por entidade (dimensão configurável; produção = 32) |
| Ramo **GMF** | Produto elemento a elemento dos embeddings → projeção linear |
| Ramo **MLP** | Concatenação dos embeddings → `Dense(64) → Dense(32)` com dropout + L2 |
| Fusão | Concatenação GMF ⊕ MLP → camada densa final → sigmoide |
| Perda | `binary_crossentropy` sobre pares positivos/negativos |
| Regularização | L2 nos embeddings + Dropout + *early stopping* por `ndcg@7` de validação |

</div>

**Por que ranking e não regressão?** Prever `Sales` normalizado produz R²≈0 e scores praticamente constantes — um modelo que não ordena o catálogo. O objetivo correto de um recomendador é *separar itens relevantes de não relevantes*, o que exige uma **perda de ranking**. Este é o ajuste que dá ao sistema o comportamento esperado de um recomendador de produção.

**Protocolo de avaliação (*leave-last-out*):** o **último** item de cada cliente é o teste e o **penúltimo** é a validação. A avaliação ranqueia o **catálogo completo**, **excluindo os itens já vistos**, e reporta:

- **NDCG@k**, **MAP@k**, **Hit-Rate@k** (métricas de ranking)
- **Precision@k**, **Recall@k**, **F1@k**
- *Baselines*: **Aleatório** (esperança analítica exata) e **Mais-populares** (ranking determinístico)

## Processo de Desenvolvimento

O desenvolvimento está documentado em notebooks executáveis que **importam o pipeline real de `src/`** (sem métricas fabricadas):

<div align="center">

| Notebook | Conteúdo |
|:--|:--|
| `01_777_Sistema_Recomendacao_Inicial.ipynb` | Análise exploratória inicial, pré-processamento e correções. |
| `02_777_Sistema_Recomendacao_Final.ipynb` | Primeira versão do modelo e geração das recomendações. |
| `03_777_Sistema_Recomendacao_Producao.ipynb` | Avaliação de ranking em produção, recomendações e *cold start*. |
| `04_777_Metodo_SHAP.ipynb` | Introdução ao SHAP e geração de relatório em PDF. |
| `05_SHAP_Avancado.ipynb` | SHAP avançado + **verificação independente** por permutação. |
| `06_777_Otimizacao_Avancada_Modelo.ipynb` | Comparação de configurações (dimensão, dropout, L2, lr). |
| `07_777_Modelo_Final_Otimizado.ipynb` | Treino final, escolha de `alpha`, registro no MLflow e recomendações. |

</div>

Os scripts em `src/scripts/` reproduzem cada etapa de forma reprodutível e são encadeados pelo `dvc.yaml`.

## Resultados (base real sc_base_01)

Métricas de **teste** (leave-last-out, catálogo completo, `k = 7`), do modelo final (`alpha = 0,0`):

<div align="center">

| Política | NDCG@7 | MAP@7 | HitRate@7 | Precision@7 | Recall@7 | F1@7 |
|:--|--:|--:|--:|--:|--:|--:|
| **Híbrido (final)** | 0,0027 | 0,0020 | 0,0052 | 0,0007 | 0,0052 | 0,0013 |
| NeuMF (neural puro) | 0,0009 | 0,0004 | 0,0026 | 0,0004 | 0,0026 | 0,0006 |
| Mais populares | 0,0027 | 0,0020 | 0,0052 | 0,0007 | 0,0052 | 0,0013 |
| Aleatório (esperança) | — | — | 0,0038 | 0,0005 | 0,0038 | — |

</div>

> O símbolo **—** indica métricas sem forma fechada no *baseline* aleatório (apenas Hit-Rate, Precision e Recall possuem esperança analítica exata).

**Seleção de hiperparâmetros** (notebook `06`):

<div align="center">

| Configuração | dim | dropout | L2 | lr | NDCG@7 val. | NDCG@7 teste |
|:--|--:|--:|--:|--:|--:|--:|
| compacto | 16 | 0,3 | 0,001 | 0,0010 | 0,00424 | **0,00428** |
| produção | 32 | 0,3 | 0,001 | 0,0010 | 0,00356 | 0,00093 |
| **regularizado** | 64 | 0,5 | 0,010 | 0,0005 | **0,00622** | 0,00222 |

</div>

A configuração **regularizado** venceu por NDCG@7 de **validação** (critério de seleção); a `compacto` teve melhor generalização em teste, mas menor NDCG@7 de validação — optou-se por não ajustar ao teste (*seleção honesta*).

**Calibração do `alpha`** do híbrido (notebook `07`):

<div align="center">

| alpha | NDCG@7 val. |
|:--|--:|
| **0,0** | **0,00478** |
| 0,1 | 0,00337 |
| 0,25 | 0,00212 |
| 0,5 | 0,00105 |
| 1,0 | 0,00214 |

</div>

**Leitura sênior dos resultados.** Nesta base esparsa de baixa cardinalidade, o *baseline* **mais-populares** é o sinal mais forte e o sinal neural **puro não o supera**; por isso o `alpha` da validação convergiu para `0,0`, degenerando o híbrido no recomendador popular. Isso **não é um defeito** — é o resultado correto de um protocolo de seleção que prioriza a validação e evita *cherry-picking* no teste. O sistema **entrega valor** (supera largamente o aleatório: Hit-Rate@7 ≈ 0,0052 vs. 0,0038) e a infraestrutura está pronta para, com mais dados, o sinal neural passar a dominar. A quantificação desse comportamento é evidenciada em [Explicabilidade (SHAP)](#explicabilidade-shap); a prova de que o **método** supera a popularidade quando há escala de dados está em [Benchmark em Base Pública Real (sc_base_03)](#benchmark-em-base-pública-real-sc_base_03).

## Benchmark em Base Pública Real (sc_base_03)

> **Aviso de honestidade.** A base desta seção é **real** — o *log de eventos* público da **sc_base_03**, com *feedback* implícito de verdade (`view`, `addtocart`, `transaction`). Ela é **anonimizada** (sem nomes de produto/cliente) e recebe **filtros explícitos de tratabilidade** (itens/usuários relevantes + amostra determinística, `src/config/sc_base_03_data.py`). Por ser dado real, **não existe "teto oracle"** (não há fator latente verdadeiro conhecido).

A sc_base_03 é um e-commerce **real**, com sinal implícito que o sistema não controla. Aqui o mesmo pipeline é executado sobre dado real de outra origem, sem qualquer ajuste de código.

<div align="center">

| Base sc_base_03 (após filtros) | Valor |
|:--|--:|
| Interações | 92.907 |
| Clientes / Produtos | 5.000 / 3.745 |
| Categorias / Subcategorias | 431 / 431 |
| Itens por cliente (média / mediana) | 10,2 / 6 |
| Densidade da matriz cliente × produto | 0,50 % |
| Filtros | itens ≥ 100 eventos · usuários ≥ 5 itens · amostra de 5.000 clientes |

</div>

Métricas de **teste** (*leave-last-out*, catálogo completo, `k = 7`):

<div align="center">

| Política | NDCG@7 | MAP@7 | HitRate@7 | Precision@7 | Recall@7 | F1@7 |
|:--|--:|--:|--:|--:|--:|--:|
| **Híbrido popularity-first (α = 1,0) — modelo final** | **0,0541** | **0,0409** | **0,0962** | **0,0137** | **0,0962** | **0,0241** |
| NeuMF puro | 0,0457 | 0,0344 | 0,0820 | 0,0117 | 0,0820 | 0,0205 |
| Mais-populares (*baseline*) | 0,0117 | 0,0085 | 0,0218 | 0,0031 | 0,0218 | 0,0055 |
| Aleatório (esperança) | — | — | 0,0019 | 0,0003 | 0,0019 | — |

</div>

**Leitura sênior.** Em dado real e anonimizado, o modelo **supera o *baseline* de popularidade em +363,5 % de NDCG@7** (0,0541 vs. 0,0117) — quase **1 em cada 10** clientes recebe o item relevante no top-7 (Hit-Rate@7 = 9,6 %, contra 2,2 % dos mais-populares e 0,19 % do aleatório). Diferente da base sc_base_01 (onde o híbrido colapsa em popularidade, `α = 0`), aqui a validação escolhe o **híbrido com personalização plena** (`α = 1,0`) — a rede de fato aprende preferências individuais. O ganho do NeuMF puro (0,0457) sobre a popularidade (0,0117) é a evidência de que **há sinal colaborativo real** capturado pelo modelo; o híbrido o reforça somando o *prior* de popularidade.

**Reprodutibilidade:** script `python -m src.scripts.benchmark_sc_base_03` (estágio DVC `benchmark_publico_sc_base_03`) · métricas em [`report/benchmark_sc_base_03_metricas.json`](report/benchmark_sc_base_03_metricas.json). Os CSVs brutos da base (`dados/sc_base_03/`, ~1 GB) são obtidos na etapa 2 do roadmap (ver `docs/ROADMAP_RECOMENDACAO_ESCALA.md`) e **não são versionados**.

## Benchmark de Escala (sc_base_04)

> **Aviso de honestidade.** A `sc_base_04` é **real**: ~14 GB de log de eventos implícitos (`view`, `cart`, `purchase`) de uma loja multicategoria. Nenhum número abaixo é estimado — todos são medidos no bruto por um estágio DVC reprodutível (`benchmark_escala_sc_base_04`).

O benchmark lê o log **uma única vez** em blocos, aplica os filtros de tratabilidade e avalia **baselines fortes** na **régua blindada** (`src/config/avaliacao_ranking.py`): leave-last-out, catálogo completo com exclusão dos itens vistos, `k = 10`, e **IC 95 % por bootstrap** sobre os usuários.

> **Nota sobre o corte `k`.** São dois cortes independentes, por desenho: o **pipeline NeuMF** (sc_base_01 e sc_base_03) usa `k = 7` — o mesmo default do serviço de inferência —, enquanto o **benchmark de escala** (sc_base_04) adota `k = 10`, propositalmente mais profundo, para expor de forma mais dura a separação entre personalização e popularidade. Os valores `k = 7` e `k = 10` **não são comparáveis entre si**.

<div align="center">

| Base sc_base_04 (após filtros) | Valor |
|:--|--:|
| Eventos lidos | 109.950.743 |
| Usuários / Produtos (bruto) | 5.316.649 / 206.876 |
| Coorte avaliada | 50.000 usuários (de 2.338.236 elegíveis) |
| Itens no catálogo candidato | 105.047 |
| Interações de treino na coorte | 1.004.004 |
| Filtros | itens ≥ 50 eventos · usuários ≥ 5 itens · coorte determinística de 50.000 |

</div>

Métricas de **teste** (*leave-last-out*, catálogo completo, `k = 10`, IC 95 % por bootstrap):

<div align="center">

| Política | NDCG@10 [IC 95 %] | MAP@10 | Hit-Rate@10 | Cobertura | Novidade |
|:--|--:|--:|--:|--:|--:|
| **Híbrido ALS + item-kNN (α = 0,5; β = 0,5) — modelo final** | **0,1257** [0,1232; 0,1280] | 0,1005 | 0,2077 | 0,282 | 11,58 |
| item-kNN (co-ocorrência) | 0,1074 [0,1052; 0,1097] | 0,0858 | 0,1778 | 0,450 | 14,12 |
| Híbrido popularity-first (α = 0,5) | 0,0947 [0,0925; 0,0968] | 0,0744 | 0,1614 | 0,013 | 10,19 |
| ALS implícito | 0,0946 [0,0924; 0,0967] | 0,0741 | 0,1618 | 0,019 | 10,58 |
| Mais-populares (*baseline*) | 0,0417 [0,0402; 0,0431] | 0,0307 | 0,0779 | 0,000 | 8,18 |
| Recência | 0,0402 [0,0387; 0,0415] | 0,0295 | 0,0755 | 0,000 | 8,31 |
| Aleatório | ~0,0000 | ~0,0000 | 0,0001 | 0,000 | 18,13 |

</div>

**Leitura sênior.** Em escala real, a **personalização vence com folga**: o modelo final (híbrido ALS + item-kNN) atinge **0,1257** de NDCG@10 — **+201 %** sobre os mais-populares e **+17 %** sobre o melhor baseline isolado (item-kNN, 0,1074). Quase **1 em cada 4,8** usuários recebe o item relevante no top-10 (Hit-Rate@10 = 20,8 %, contra 7,8 % da popularidade e 0,01 % do aleatório). Os intervalos de confiança **não se sobrepõem** entre o modelo final e qualquer baseline — o ganho é estatisticamente sólido, não ruído.

**Como se chegou ao modelo final.** A régua blindada primeiro **expôs** que o item-kNN de co-ocorrência superava o híbrido popularity-first baseado em ALS (0,1074 vs. 0,0947, ≈ +12 %). Em vez de maquiar o resultado, o desenho foi corrigido: o **híbrido ALS + item-kNN** soma o prior de popularidade ao sinal latente (α) e à co-ocorrência (β), com **os dois pesos escolhidos na validação** (α = 0,5; β = 0,5). O resultado é o melhor dos dois mundos: NDCG@10 de **0,1257**, acima do item-kNN puro, preservando boa cobertura (28,2 % do catálogo). Esse ciclo — **medir contra o melhor baseline → revelar a fraqueza → redesenhar → medir de novo** — é o que caracteriza uma avaliação de nível sênior.

**Reprodutibilidade:** script `python -m src.scripts.benchmark_sc_base_04` (estágio DVC `benchmark_escala_sc_base_04`) · métricas em [`report/benchmark_sc_base_04_metricas.json`](report/benchmark_sc_base_04_metricas.json). Os CSVs brutos (`dados/sc_base_04/`, ~14 GB) **não são versionados**.

## Explicabilidade (SHAP)

O SHAP é aplicado ao *score* da NeuMF para medir a contribuição de cada entrada. Como a rede é multi-entrada (IDs), a análise é feita sobre os quatro campos (cliente, produto, categoria, subcategoria). Os resultados (notebook `05`) são consistentes entre **três evidências independentes**:

<div align="center">

| Entrada | Importância SHAP | \|Δ score\| ao permutar | \|W\|máx. do embedding |
|:--|--:|--:|--:|
| Subcategoria | 0,00729 | 0,008615 | 0,141193 |
| Categoria | 0,00094 | 0,001145 | 0,115556 |
| Cliente | 0,00000 | 0,000000 | 0,000000 |
| Produto | 0,00000 | 0,000000 | 0,000000 |

</div>

As três colunas convergem: **categoria e subcategoria** concentram a informação; os embeddings de **cliente e produto colapsam a zero** (`|W|max ≈ 0`), efeito da regularização L2 sobre entidades de altíssima cardinalidade e pouquíssimas observações por entidade. Essa é exatamente a razão técnica pela qual o híbrido **popularity-first** é o desenho correto aqui — o sinal confiável está na popularidade e nos atributos de categoria, não nas interações individuais. As figuras estão em `report/figuras/`.

## Ambiente MLOps

### uv — ambiente reprodutível
O `pyproject.toml` é a **fonte única** do ambiente; o `uv.lock` fixa todas as versões resolvidas (reprodutibilidade total) e o `requirements.txt` é **exportado** do lock para compatibilidade com ferramentas legadas.

### DVC + DagsHub — versionamento de dados e modelos
O *remote* do DVC é o **DagsHub deste projeto** (`https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.dvc`). O pipeline é declarado em `dvc.yaml` com sete estágios:

| Estágio | Comando | Saídas versionadas |
|:--|:--|:--|
| `perfil_dados` | `python -m src.scripts.perfil_dados` | `report/perfil_sc_base_01.json` |
| `preparacao_dados` | `python -m src.scripts.preparar_dados` | `dados/sc_base_01/sc_base_01_tratado.parquet` |
| `treinamento_modelo` | `python -m src.scripts.otimizacao_modelo` | `models/best_model_recomendacao_otimizado.keras`, `models/hybrid_config_otimizado.pkl`, `models/*_encoder_otimizado.pkl` |
| `treinamento_ensemble` | `python -m src.scripts.modelo_ensemble` | `models/ensemble/` |
| `benchmark_publico_sc_base_03` | `python -m src.scripts.benchmark_sc_base_03` | `dados/sc_base_03/benchmark_sc_base_03.parquet` |
| `perfil_sc_base_04` | `python -m src.scripts.perfil_sc_base_04` | `report/perfil_sc_base_04.json` |
| `benchmark_escala_sc_base_04` | `python -m src.scripts.benchmark_sc_base_04` | `report/benchmark_sc_base_04_metricas.json` |

### MLflow — tracking e model registry
- Backend **local em SQLite** (`sqlite:///<raiz>/mlflow.db`) por padrão — 100 % offline e recomendado pelo MLflow 3 (o antigo *file store* `mlruns/` está em *maintenance mode*).
- *Opt-in* remoto via variável **própria** do projeto (`RECOMENDACAO_MLFLOW_TRACKING_URI`) para não herdar credenciais de outro repositório.
- Modelo registrado como **`recomendador_neumf`** e consumido por **alias** (`models:/recomendador_neumf@champion`).

### GitHub Actions — CI/CD
Pipeline em `.github/workflows/ci.yml` com `uv sync --frozen`, **Ruff** (lint), **Black** (formato) e **pytest** (testes).

## Estrutura do Repositório

```
Sistema_Recomendacao_Arquitetura-MLOps/
├── dados/                     # Uma pasta por base (sc_base_01, sc_base_03..05) — DVC
├── models/                    # Modelo, encoders e config do híbrido — DVC
├── notebooks/                 # 01..07 — narrativa executável do projeto
├── src/
│   ├── config/
│   │   ├── model_utils.py     # NeuMF, híbrido, métricas de ranking, baselines
│   │   ├── avaliacao_ranking.py # Régua de avaliação (métricas por usuário, IC 95%, baselines escaláveis)
│   │   ├── paths.py           # Caminhos canônicos do projeto
│   │   ├── sc_base_03_data.py # Preparo da base pública sc_base_03 (benchmark real)
│   │   ├── auxiliares_ml.py   # Utilitários de preparação de dados (downcast)
│   │   ├── mlflow_tracking.py # Setup do MLflow (SQLite/DagsHub) e registro
│   │   ├── dagshub_config.py  # Integração DagsHub ↔ MLflow
│   │   └── experiment_tracker.py # Registro de experimentos
│   └── scripts/
│       ├── perfil_dados.py        # Auditoria de qualidade da base bruta
│       ├── perfil_sc_base_04.py   # Auditoria do log de eventos em escala (sc_base_04)
│       ├── preparar_dados.py      # CSV → Parquet tratado
│       ├── otimizacao_modelo.py   # Treino do híbrido + métricas + MLflow
│       ├── modelo_ensemble.py     # Ensemble (estágio complementar)
│       ├── benchmark_sc_base_03.py # Benchmark em base pública real (sc_base_03)
│       └── benchmark_sc_base_04.py # Benchmark de escala (sc_base_04) com a régua blindada
├── serving/                   # Serviço de inferência (FastAPI): app, service, schemas
├── tests/                     # pytest (utilidades do modelo, sistema e API)
├── report/                    # Figuras SHAP e métricas dos benchmarks (JSON)
├── docs/                      # Guias (OPTIMIZATION_GUIDE.md, SYNC_GUIDE.md, ROADMAP_RECOMENDACAO_ESCALA.md)
├── .github/workflows/ci.yml   # CI/CD
├── Dockerfile · .dockerignore # Empacotamento do serviço (multi-stage, non-root)
├── dvc.yaml · .dvc/           # Pipeline e remote DagsHub
├── pyproject.toml · uv.lock   # Ambiente reprodutível (uv)
├── .env.example               # Modelo de credenciais (copiar para .env)
└── README.md · LICENSE.md
```

## Instalação e Uso

Pré-requisitos: **Python 3.11**, **uv** (instalação: <https://docs.astral.sh/uv/>) e Git.

```bash
# 1. Clonar o repositório
git clone https://github.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.git
cd Sistema_Recomendacao_Arquitetura-MLOps

# 2. Criar o ambiente a partir do lock (reprodutível)
uv sync --frozen

# 3. Baixar dados e modelos versionados (DVC/DagsHub)
uv run dvc pull

# 4. Configurar credenciais (opcional, para DagsHub/MLflow remoto)
cp .env.example .env      # preencha DAGSHUB_USER_TOKEN
```

**Reproduzir o pipeline** (gera dados tratados, treina e salva os artefatos):

```bash
uv run dvc repro
```

**Abrir os notebooks:**

```bash
uv run jupyter lab
```

**MLflow UI (tracking local):**

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Testes

```bash
uv run python -m pytest tests/ -v
```

A suíte cobre as utilidades do modelo (criação da NeuMF, prior de popularidade, combinação de scores, baselines), o comportamento do sistema de recomendação e a **API de inferência** (`tests/test_api.py`, com uma fixture mínima — não depende dos artefatos do DVC).

## Serviço de Inferência (API)

O modelo treinado é exposto como um serviço REST de **referência** (`serving/`), pronto para ser empacotado e entregue a um time de produção. O serviço é **agnóstico de base**: a base servida é declarada em `DATA_PATH` (ex.: `sc_base_01`) e o papel do conjunto em `MODEL_VARIANT` — o conjunto entregue hoje é o de `sc_base_01`, rotulado como **`baseline`** (teste de fumaça de infraestrutura), não como modelo principal da API.

- **`GET /health`** — estado do serviço, base servida (`base_code`), papel do conjunto (`variant`) e artefatos carregados;
- **`POST /recommend`** — top-k com `ranking_mode` explícito (`popularity` ou `hybrid`) e **SHAP aditivo sobre o score efetivamente usado no ranking** (cliente, produto, categoria, subcategoria). Clientes sem histórico (*cold start*) recebem o *fallback* por popularidade, com `cold_start = true` e explicação vazia. O `alpha` é selecionado por validação; nesta base sc_base_01, o modo atual é `popularity`.

O serviço carrega os artefatos (`models/` e o parquet indicado por `DATA_PATH`, versionados no DVC) **uma única vez** no *startup*; se eles não estiverem disponíveis, `/recommend` responde **503** em vez de falhar. Sem `DATA_PATH` o serviço não assume nenhuma base e recusa o *startup*.

**Subir localmente** (sem Docker) — `DATA_PATH` é obrigatório:

```bash
export DATA_PATH=dados/sc_base_01/sc_base_01_tratado.parquet
uv run uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

**Build e execução via Docker** (rode `uv run dvc pull` antes, para que os artefatos existam no contexto de build):

```bash
docker build -t sistema-recomendacao:latest .
docker run --rm -p 8000:8000 sistema-recomendacao:latest
```

**Exemplo de requisição:**

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Cliente Exemplo", "top_k": 7, "explain": true}'
```

A documentação interativa (OpenAPI/Swagger) fica em <http://localhost:8000/docs>.

> **Nota de honestidade:** trata-se de um serviço de *referência* reprodutível — **"pronto para produção" ≠ "implantado em produção"**. Não há, por escopo, autenticação, cache distribuído nem orquestração (Kubernetes).

## Observações Importantes

- **Cold start:** a função de recomendação trata **novos clientes** (sem histórico) acionando um *fallback* de popularidade, garantindo recomendações válidas para qualquer consumidor.
- **Reprodutibilidade:** todas as funções usam sementes fixas (`RANDOM_SEED = 42`), o ambiente é congelado pelo `uv.lock` e os artefatos são versionados pelo DVC.
- **Credenciais isoladas:** o rastreamento usa variáveis **próprias do projeto** para não herdar configuração de outros repositórios.
- **Limitação honesta:** o desempenho é limitado pelo tamanho e pela esparsidade da base; o pipeline de ranking, a metodologia de avaliação e a explicabilidade estão prontos para escalar com mais dados.
- **Bases e papéis:** `sc_base_01` é o **teste de fumaça de infraestrutura** (ETL, DVC, MLflow, SHAP, cold start) e expõe sua limitação honesta; `sc_base_03` prova que o **mesmo pipeline generaliza para dado real de outra origem** (+363,5 % de NDCG@7 sobre a popularidade, com personalização plena `α = 1,0`); `sc_base_04` é o **benchmark principal de escala** — em 109,95 milhões de eventos, o modelo final (híbrido ALS + item-kNN) atinge NDCG@10 de 0,1257, **+201 % sobre a popularidade** e **+17 % sobre o melhor baseline isolado** (ver [Benchmark de Escala (sc_base_04)](#benchmark-de-escala-sc_base_04)); `sc_base_05` é o **domínio cruzado**, reservado para validar generalização.

## Licença

Este modelo está licenciado sob a **Licença MIT**. Consulte [LICENSE.md](LICENSE.md).

## Contato

- **Nome:** Flávio Henrique Barbosa
- **LinkedIn:** [Flávio Henrique Barbosa](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
- **E-mail:** flaviohenriquehb777@outlook.com
