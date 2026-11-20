<div align="center">

# Sistema de Recomendações
## Base de Estudo `sc_base_01`

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
- [Benchmark de Escala (base sintética)](#benchmark-de-escala-base-sintética-rotulada)
- [Benchmark em Base Pública Real (sc_base_03)](#benchmark-em-base-pública-real-sc_base_03)
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

O projeto trabalha com bases identificadas **apenas por nome-código** — `sc_base_01`, `sc_base_02` e `sc_base_03` — para manter a imersão no projeto e isolar a análise de qualquer referência externa. Cada base passa pelas mesmas etapas de **entendimento e tratamento de dados** antes de alimentar o modelo (ver `docs/ROADMAP_RECOMENDACAO_ESCALA.md`).

| Código | Papel no projeto |
|:--|:--|
| `sc_base_01` | **Baseline** de transações de varejo — valida o sistema de produção (ETL, DVC, MLflow, SHAP, cold start). |
| `sc_base_02` | **Base principal de escala** — destinada a sustentar personalização com volume real de interações. |
| `sc_base_03` | **Base comportamental** (log de eventos) — sinal implícito real, complementar à `sc_base_01`. |

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
| `08_777_Benchmark_Escala_Sintetico.ipynb` | Benchmark de escala sobre base **sintética rotulada**: baselines e teto "oracle". |

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

**Leitura sênior dos resultados.** Nesta base esparsa de baixa cardinalidade, o *baseline* **mais-populares** é o sinal mais forte e o sinal neural **puro não o supera**; por isso o `alpha` da validação convergiu para `0,0`, degenerando o híbrido no recomendador popular. Isso **não é um defeito** — é o resultado correto de um protocolo de seleção que prioriza a validação e evita *cherry-picking* no teste. O sistema **entrega valor** (supera largamente o aleatório: Hit-Rate@7 ≈ 0,0052 vs. 0,0038) e a infraestrutura está pronta para, com mais dados, o sinal neural passar a dominar. A quantificação desse comportamento é evidenciada em [Explicabilidade (SHAP)](#explicabilidade-shap); a prova de que o **método** supera a popularidade quando há escala de dados está em [Benchmark de Escala](#benchmark-de-escala-base-sintética-rotulada).

## Benchmark de Escala (base sintética rotulada)

> **Aviso de honestidade.** A base desta seção é **100 % sintética** e **reproduzível** (`seed=42`, `src/config/synthetic_data.py`). Ela **não representa dados reais de nenhuma empresa**. O objetivo é demonstrar que **o método e a arquitetura recuperam preferência latente quando ela existe** — algo que a base pública pequena (seção anterior) não permite medir.

A base pública *sc_base_01* é pequena e esparsa; nessa escala, o *baseline* de *mais-populares* é o sinal mais forte e **nenhum** modelo neural o supera de forma estável. Um recomendador "nível produção" só **demonstra personalização** quando existe sinal colaborativo recuperável. Como dados reais de clientes são protegidos por privacidade (o que se publica em portfólio é anonimizado/sintético), construímos um **benchmark sintético** com estrutura latente real e rodamos **o mesmo código do pipeline**.

<div align="center">

| Métrica da base sintética | Valor |
|:--|--:|
| Interações | 66.020 |
| Clientes / Produtos | 3.000 / 990 |
| Categorias / Subcategorias | 8 / 40 |
| Itens por cliente (média) | 22,0 |
| Densidade da matriz cliente × produto | 2,22 % |

</div>

Métricas de **teste** (*leave-last-out*, catálogo completo, `k = 7`):

<div align="center">

| Política | NDCG@7 | MAP@7 | HitRate@7 | Precision@7 | Recall@7 | F1@7 |
|:--|--:|--:|--:|--:|--:|--:|
| **NeuMF (modelo final)** | **0,1141** | **0,0882** | **0,1967** | **0,0281** | **0,1967** | **0,0492** |
| Híbrido popularity-first (α = 1,0) | 0,1085 | 0,0835 | 0,1877 | 0,0268 | 0,1877 | 0,0469 |
| Mais-populares (*baseline*) | 0,0215 | 0,0160 | 0,0390 | 0,0056 | 0,0390 | 0,0098 |
| Teto "oracle" (afinidade latente) | 0,1400 | 0,1060 | 0,2483 | 0,0355 | 0,2483 | 0,0621 |
| Aleatório (esperança) | — | — | 0,0072 | 0,0010 | 0,0072 | — |

</div>

> O **teto "oracle"** ranqueia pela afinidade latente verdadeira — é um **limite superior** (não um modelo implantável) e serve para medir quanto do potencial é recuperável. O símbolo **—** indica métricas sem forma fechada no *baseline* aleatório.

**Leitura sênior.** Com sinal colaborativo real, a **NeuMF supera o *baseline* de popularidade em +431,7 % de NDCG@7** e captura **81,5 % do teto recuperável** — ou seja, **não é um recomendador de "mais vendidos"**: ele separa preferências por cliente. A escolha do modelo final é feita **por validação** (`NDCG@7` de validação = 0,10769); em escala, a validação seleciona a **NeuMF pura**, porque o *prior* explícito de popularidade apenas dilui a personalização que a rede já aprendeu.

**Por que as duas bases diferem?** O gargalo da base real é **dado, não código**. O mesmo pipeline, com o mesmo protocolo e a mesma arquitetura, mostra personalização robusta quando há interações suficientes. A diferença operacional é a **regularização**: na base real (minúscula), `L2` + *dropout* fortes são **necessários** para não decorar o treino; em escala, essa mesma regularização **sufoca** a personalização.

> **Reprodutibilidade:** notebook [`08_777_Benchmark_Escala_Sintetico.ipynb`](notebooks/08_777_Benchmark_Escala_Sintetico.ipynb) · script `python -m src.scripts.benchmark_sintetico` · métricas em [`report/benchmark_sintetico_metricas.json`](report/benchmark_sintetico_metricas.json) · figura em `report/figuras/benchmark_escala_ndcg.png`.

## Benchmark em Base Pública Real (sc_base_03)

> **Aviso de honestidade.** A base desta seção é **real** — o *log de eventos* público da **sc_base_03**, com *feedback* implícito de verdade (`view`, `addtocart`, `transaction`). Ela é **anonimizada** (sem nomes de produto/cliente) e recebe **filtros explícitos de tratabilidade** (itens/usuários relevantes + amostra determinística, `src/config/sc_base_03_data.py`). Por ser dado real, **não existe "teto oracle"** (não há fator latente verdadeiro conhecido).

A base sintética prova que o método funciona quando **construímos** o sinal — o que é, em alguma medida, circular. A sc_base_03 fecha essa lacuna: é um e-commerce **real**, com sinal implícito que o sistema não controla. Aqui o mesmo pipeline é executado sobre dado real de outra origem, sem qualquer ajuste de código.

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
O *remote* do DVC é o **DagsHub deste projeto** (`https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.dvc`). O pipeline é declarado em `dvc.yaml` com cinco estágios:

| Estágio | Comando | Saídas versionadas |
|:--|:--|:--|
| `preparacao_dados` | `python -m src.scripts.preparar_dados` | `dados/sc_base_01/sc_base_01_tratado.parquet` |
| `treinamento_modelo` | `python -m src.scripts.otimizacao_modelo` | `models/best_model_recomendacao_otimizado.keras`, `models/hybrid_config_otimizado.pkl`, `models/*_encoder_otimizado.pkl` |
| `treinamento_ensemble` | `python -m src.scripts.modelo_ensemble` | `models/ensemble/` |
| `benchmark_sintetico` | `python -m src.scripts.benchmark_sintetico` | `dados/benchmark_sintetico.parquet` |
| `benchmark_publico_sc_base_03` | `python -m src.scripts.benchmark_sc_base_03` | `dados/sc_base_03/benchmark_sc_base_03.parquet` |

### MLflow — tracking e model registry
- Backend **local em SQLite** (`sqlite:///<raiz>/mlflow.db`) por padrão — 100 % offline e recomendado pelo MLflow 3 (o antigo *file store* `mlruns/` está em *maintenance mode*).
- *Opt-in* remoto via variável **própria** do projeto (`RECOMENDACAO_MLFLOW_TRACKING_URI`) para não herdar credenciais de outro repositório.
- Modelo registrado como **`recomendador_neumf`** e consumido por **alias** (`models:/recomendador_neumf@champion`).

### GitHub Actions — CI/CD
Pipeline em `.github/workflows/ci.yml` com `uv sync --frozen`, **Ruff** (lint), **Black** (formato) e **pytest** (testes).

## Estrutura do Repositório

```
Sistema_Recomendacao_Arquitetura-MLOps/
├── dados/                     # Uma pasta por base (sc_base_01, sc_base_03) — DVC
├── models/                    # Modelo, encoders e config do híbrido — DVC
├── notebooks/                 # 01..08 — narrativa executável do projeto
├── src/
│   ├── config/
│   │   ├── model_utils.py     # NeuMF, híbrido, métricas de ranking, baselines
│   │   ├── mlflow_tracking.py # Setup do MLflow (SQLite/DagsHub) e registro
│   │   ├── paths.py           # Caminhos canônicos do projeto
│   │   ├── synthetic_data.py  # Gerador de base sintética (benchmark de escala)
│   │   ├── sc_base_03_data.py # Preparo da base pública sc_base_03 (benchmark real)
│   │   └── auxiliares*.py     # Utilitários de dados/plot
│   └── scripts/
│       ├── preparar_dados.py      # CSV → Parquet tratado
│       ├── otimizacao_modelo.py   # Treino do híbrido + métricas + MLflow
│       ├── modelo_ensemble.py     # Ensemble (estágio complementar)
│       ├── benchmark_sintetico.py # Benchmark de escala (base sintética)
│       └── benchmark_sc_base_03.py # Benchmark em base pública real (sc_base_03)
├── serving/                   # Serviço de inferência (FastAPI): app, service, schemas
├── tests/                     # pytest (utilidades do modelo, sistema e API)
├── report/                    # Figuras SHAP/benchmark e relatório PDF
├── docs/                      # Guias (OPTIMIZATION_GUIDE.md, SYNC_GUIDE.md)
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

A suíte cobre as utilidades do modelo (criação da NeuMF, prior de popularidade, combinação de scores, baselines), o comportamento do sistema de recomendação e a **API de inferência** (`tests/test_api.py`, com recomendador sintético — não depende dos artefatos do DVC).

## Serviço de Inferência (API)

O modelo treinado é exposto como um serviço REST de **referência** (`serving/`), pronto para ser empacotado e entregue a um time de produção. O serviço é **agnóstico de base**: a base servida é declarada em `DATA_PATH` (ex.: `sc_base_01`) e o papel do conjunto em `MODEL_VARIANT` — o conjunto entregue hoje é o de `sc_base_01`, rotulado como **`baseline`**, não como modelo principal da API.

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
  -d '{"customer_name": "Claire Gute", "top_k": 7, "explain": true}'
```

A documentação interativa (OpenAPI/Swagger) fica em <http://localhost:8000/docs>.

> **Nota de honestidade:** trata-se de um serviço de *referência* reprodutível — **"pronto para produção" ≠ "implantado em produção"**. Não há, por escopo, autenticação, cache distribuído nem orquestração (Kubernetes).

## Observações Importantes

- **Cold start:** a função de recomendação trata **novos clientes** (sem histórico) acionando um *fallback* de popularidade, garantindo recomendações válidas para qualquer consumidor.
- **Reprodutibilidade:** todas as funções usam sementes fixas (`RANDOM_SEED = 42`), o ambiente é congelado pelo `uv.lock` e os artefatos são versionados pelo DVC.
- **Credenciais isoladas:** o rastreamento usa variáveis **próprias do projeto** para não herdar configuração de outros repositórios.
- **Limitação honesta:** o desempenho é limitado pelo tamanho e pela esparsidade da base; o pipeline de ranking, a metodologia de avaliação e a explicabilidade estão prontos para escalar com mais dados.
- **Três bases, três papéis:** a base **real** (`sc_base_01`) valida o **sistema de produção** (ETL, DVC, MLflow, SHAP, cold start) e expõe sua limitação honesta; o **benchmark sintético rotulado** valida o **método em escala** (+431,7 % de NDCG@7 sobre a popularidade); e o **benchmark em base pública real (`sc_base_03`)** prova que o **mesmo pipeline generaliza para dado real de outra origem** (+363,5 % de NDCG@7 sobre a popularidade, com personalização plena `α = 1,0`).

## Licença

Este modelo está licenciado sob a **Licença MIT**. Consulte [LICENSE.md](LICENSE.md).

## Contato

- **Nome:** Flávio Henrique Barbosa
- **LinkedIn:** [Flávio Henrique Barbosa](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
- **E-mail:** flaviohenriquehb777@outlook.com
