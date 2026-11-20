"""
Utilitários de modelagem do sistema de recomendação.

Abordagem
---------
O recomendador é uma rede de filtragem colaborativa neural no estilo **NeuMF**
(GMF + MLP, He et al., 2017) treinada com **objetivo de ranking**: trata-se de
uma classificação binária de interação (o cliente interage / não interage com o
produto), com **amostragem de negativos**, em vez de regressão sobre ``Sales``.

Por que não regressão?
    Prever ``Sales`` normalizado produz R²≈0 e um modelo que praticamente não
    ordena o catálogo (scores quase constantes). O objetivo correto de um
    sistema de recomendação é *separar itens relevantes de não relevantes* — ou
    seja, ranquear —, o que exige uma perda de ranking (aqui, entropia cruzada
    binária sobre pares positivos/negativos).

Avaliação
---------
Protocolo **leave-last-out** (o último item de cada cliente é o teste): ranqueando
o **catálogo completo** e **excluindo os itens já vistos** pelo cliente:

    * NDCG@k, MAP@k, Hit-Rate@k (métricas de ranking)
    * Precision@k, Recall@k, F1@k
    * Baselines: aleatório e mais-populares (Most-Popular)

O penúltimo item de cada cliente é usado como **validação** (early stopping sobre
o NDCG@k de validação). O modelo final é um **híbrido popularity-first**:

    score = z(log1p(popularidade)) + alpha * z(NeuMF)

isto é, o prior de popularidade (contado no treino) é o sinal-base e o sinal
personalizado do NeuMF entra como um ajuste ponderado por ``alpha``, escolhido na
validação. ``alpha=0`` degenera para o recomendador puramente popular — que,
nesta base (≈12 itens por cliente num catálogo de ~1.900), é o baseline mais
forte.

Todas as funções são determinísticas (sementes fixas) para garantir
reprodutibilidade.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers

RANDOM_SEED = 42

# Alvo k padrão do pipeline NeuMF (07 recomendações, conforme o escopo do projeto).
TOP_K = 7


# ---------------------------------------------------------------------------
# Catálogo de itens
# ---------------------------------------------------------------------------
def build_item_catalog(
    df: pd.DataFrame,
    product_encoder,
    category_encoder,
    subcategory_encoder,
    product_col: str = "Product ID",
    category_col: str = "Category",
    subcategory_col: str = "Sub-Category",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói o catálogo de itens alinhado ao índice do ``LabelEncoder`` de produtos.

    Como categoria/subcategoria são determinísticas por produto, usamos o valor
    modal de cada produto para poder pontuar um produto sem depender do contexto
    de uma transação específica.

    Returns:
        (product_ids, category_ids, subcategory_ids), todos ``int32`` e com
        comprimento igual ao número de produtos únicos.
    """
    tmp = pd.DataFrame(
        {
            "p": product_encoder.transform(df[product_col]),
            "c": category_encoder.transform(df[category_col]),
            "s": subcategory_encoder.transform(df[subcategory_col]),
        }
    )
    ref = tmp.groupby("p").agg(
        c=("c", lambda x: x.mode().iloc[0]),
        s=("s", lambda x: x.mode().iloc[0]),
    )
    product_ids = np.arange(len(product_encoder.classes_), dtype="int32")
    category_ids = ref["c"].reindex(product_ids).to_numpy(dtype="int32")
    subcategory_ids = ref["s"].reindex(product_ids).to_numpy(dtype="int32")
    return product_ids, category_ids, subcategory_ids


# ---------------------------------------------------------------------------
# Divisão treino / validação / teste (leave-last-out)
# ---------------------------------------------------------------------------
def train_validation_test_split(
    df: pd.DataFrame,
    customer_encoder,
    product_encoder,
    customer_col: str = "Customer Name",
    product_col: str = "Product ID",
    date_col: str = "Order Date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide as interações (cliente, produto) em treino, validação e teste.

    No protocolo **leave-last-out** (padrão em filtragem colaborativa implícita):
    o **último** item distinto de cada cliente vai para o **teste**, o
    **penúltimo** para a **validação** e os demais para o **treino**. Clientes
    com menos de 3 itens distintos são descartados (não há histórico para
    treinar nem itens para validar/testar).

    Returns:
        (train, val, test) como DataFrames com colunas ``c`` (cliente
        codificado) e ``p`` (produto codificado).
    """
    d = df[[customer_col, product_col]].copy()
    if date_col in df.columns:
        d["_ordem"] = (
            pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            .astype("int64")
            .to_numpy()
        )
    else:
        d["_ordem"] = np.arange(len(d), dtype="int64")

    d["c"] = customer_encoder.transform(d[customer_col]).astype("int64")
    d["p"] = product_encoder.transform(d[product_col]).astype("int64")

    # Interação mais recente de cada par (cliente, produto).
    pares = (
        d.sort_values("_ordem")
        .drop_duplicates(subset=["c", "p"], keep="last")
        .sort_values("_ordem")
    )
    # 0 = último item -> teste; 1 = penúltimo -> validação; >1 -> treino.
    pares["_rank"] = pares.groupby("c").cumcount(ascending=False)

    test = pares[pares["_rank"] == 0][["c", "p"]].reset_index(drop=True)
    val = pares[pares["_rank"] == 1][["c", "p"]].reset_index(drop=True)
    train = pares[pares["_rank"] > 1][["c", "p"]].reset_index(drop=True)

    clientes_validos = set(train["c"].unique())
    val = val[val["c"].isin(clientes_validos)].reset_index(drop=True)
    test = test[test["c"].isin(clientes_validos)].reset_index(drop=True)
    return train, val, test


# ---------------------------------------------------------------------------
# Amostragem de negativos
# ---------------------------------------------------------------------------
def _seen_products_by_customer(pairs: pd.DataFrame) -> dict[int, set]:
    return pairs.groupby("c")["p"].apply(set).to_dict()


def _sample_negative_products(
    customers: np.ndarray, seen: dict[int, set], num_products: int, rng: np.random.Generator
) -> np.ndarray:
    """Sorteia um produto não visto para cada cliente (com re-sorteio em colisões)."""
    negatives = rng.integers(0, num_products, size=len(customers))
    for _ in range(10):
        colisoes = np.fromiter(
            (neg in seen.get(cust, set()) for cust, neg in zip(customers, negatives)),
            dtype=bool,
            count=len(customers),
        )
        if not colisoes.any():
            break
        negatives[colisoes] = rng.integers(0, num_products, size=int(colisoes.sum()))
    return negatives


def build_training_pairs(
    train_pairs: pd.DataFrame,
    num_products: int,
    n_negatives: int = 8,
    seed: int = RANDOM_SEED,
    exclude_pairs: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monta o conjunto de treino (positivos + negativos) para a perda de ranking.

    Para cada interação observada são sorteados ``n_negatives`` produtos que o
    cliente **não** consumiu — a rede aprende a ordenar positivos acima de
    negativos.

    Args:
        exclude_pairs: interações adicionais que **não** devem ser sorteadas
            como negativas (ex.: ao montar a validação, exclui também os itens
            de treino do cliente, para não ensinar o modelo a rebaixar um item
            que ele de fato consumiu).

    Returns:
        (customers, products, labels) — ``labels`` é 1 para positivos e 0 para
        negativos.
    """
    rng = np.random.default_rng(seed)
    seen = _seen_products_by_customer(train_pairs)
    if exclude_pairs is not None:
        for cliente, itens in _seen_products_by_customer(exclude_pairs).items():
            seen.setdefault(cliente, set()).update(itens)
    customers = train_pairs["c"].to_numpy(dtype="int64")
    positives = train_pairs["p"].to_numpy(dtype="int64")

    negatives = [
        _sample_negative_products(customers, seen, num_products, rng)
        for _ in range(n_negatives)
    ]

    train_customers = np.concatenate([customers] * (n_negatives + 1))
    train_products = np.concatenate([positives, *negatives])
    labels = np.concatenate(
        [np.ones(len(positives), dtype="float32"), np.zeros(len(positives) * n_negatives, dtype="float32")]
    )

    # Embaralha para não deixar todos os positivos no início de cada lote.
    perm = rng.permutation(len(train_customers))
    return train_customers[perm], train_products[perm], labels[perm]


# ---------------------------------------------------------------------------
# Prior de popularidade (sinal-base do híbrido)
# ---------------------------------------------------------------------------
def compute_popularity_prior(train_pairs: pd.DataFrame, num_products: int) -> np.ndarray:
    """
    Conta quantas vezes cada produto aparece no treino (prior de popularidade).

    Returns:
        Vetor de comprimento ``num_products`` alinhado ao índice codificado dos
        produtos; posições sem interação ficam em zero.
    """
    contagens = train_pairs["p"].value_counts()
    prior = np.zeros(num_products, dtype="float64")
    prior[contagens.index.to_numpy(dtype="int64")] = contagens.to_numpy(dtype="float64")
    return prior


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def create_ranking_model(
    num_customers: int,
    num_products: int,
    num_categories: int,
    num_subcategories: int,
    embedding_dim: int = 32,
    l2_strength: float = 1e-3,
    dropout_rate: float = 0.3,
    hidden_units: Sequence[int] = (64, 32),
    learning_rate: float = 1e-3,
    seed: int = RANDOM_SEED,
) -> tf.keras.Model:
    """
    Cria o recomendador NeuMF (GMF + MLP) para classificação binária de interação.

    A saída é uma probabilidade (sigmoid); a perda é ``binary_crossentropy``.

    Os defaults (``embedding_dim=32``, ``l2_strength=1e-3``, ``dropout_rate=0.3``)
    foram escolhidos por validação: em bases de varejo muito esparsas (~12 itens
    por cliente em um catálogo de milhares), embeddings grandes com pouca
    regularização decoram o treino e não generalizam. Regularização forte é o que
    faz o modelo superar os baselines de popularidade e co-ocorrência.
    """
    tf.keras.utils.set_random_seed(seed)
    regularizer = regularizers.l2(l2_strength) if l2_strength else None

    customer_input = layers.Input(shape=(1,), name="customer_input")
    product_input = layers.Input(shape=(1,), name="product_input")
    category_input = layers.Input(shape=(1,), name="category_input")
    subcategory_input = layers.Input(shape=(1,), name="subcategory_input")

    customer_emb = layers.Embedding(
        num_customers, embedding_dim, embeddings_regularizer=regularizer, name="customer_embedding"
    )(customer_input)
    product_emb = layers.Embedding(
        num_products, embedding_dim, embeddings_regularizer=regularizer, name="product_embedding"
    )(product_input)
    category_emb = layers.Embedding(
        num_categories, embedding_dim // 2, embeddings_regularizer=regularizer, name="category_embedding"
    )(category_input)
    subcategory_emb = layers.Embedding(
        num_subcategories, embedding_dim // 2, embeddings_regularizer=regularizer, name="subcategory_embedding"
    )(subcategory_input)

    customer_vec = layers.Flatten(name="customer_flatten")(customer_emb)
    product_vec = layers.Flatten(name="product_flatten")(product_emb)
    category_vec = layers.Flatten(name="category_flatten")(category_emb)
    subcategory_vec = layers.Flatten(name="subcategory_flatten")(subcategory_emb)

    # Ramo GMF: interação element-wise cliente × produto (forte indutiva p/ ranking).
    gmf = layers.Multiply(name="gmf")([customer_vec, product_vec])

    # Ramo MLP: combina os embeddings com contexto de categoria/subcategoria.
    mlp = layers.Concatenate(name="mlp_concat")(
        [customer_vec, product_vec, category_vec, subcategory_vec]
    )
    mlp = layers.Dense(hidden_units[0], activation="relu", kernel_regularizer=regularizer, name="mlp_dense_1")(mlp)
    mlp = layers.Dropout(dropout_rate, name="mlp_dropout_1")(mlp)
    mlp = layers.Dense(hidden_units[1], activation="relu", kernel_regularizer=regularizer, name="mlp_dense_2")(mlp)

    merged = layers.Concatenate(name="neumf_concat")([gmf, mlp])
    output = layers.Dense(1, activation="sigmoid", name="output")(merged)

    model = tf.keras.Model(
        [customer_input, product_input, category_input, subcategory_input], output, name="recomendador_neumf"
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_ranking_model(
    model: tf.keras.Model,
    train_customers: np.ndarray,
    train_products: np.ndarray,
    categories: np.ndarray,
    subcategories: np.ndarray,
    labels: np.ndarray,
    validation_data=None,
    epochs: int = 40,
    batch_size: int = 256,
    callbacks: Sequence[tf.keras.callbacks.Callback] | None = None,
    verbose: int = 2,
) -> tf.keras.callbacks.History:
    """
    Treina o modelo de ranking.

    O early stopping é responsabilidade do ``RankingCheckpoint`` (informado em
    ``callbacks``), que monitora diretamente o NDCG@k de validação.
    """
    inputs = [
        train_customers.reshape(-1, 1),
        train_products.reshape(-1, 1),
        categories.reshape(-1, 1),
        subcategories.reshape(-1, 1),
    ]
    return model.fit(
        inputs,
        labels,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=list(callbacks) if callbacks else [],
        verbose=verbose,
    )


def to_model_inputs(
    customers: np.ndarray,
    products: np.ndarray,
    catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list[np.ndarray]:
    """Converte (clientes, produtos) nos 4 arrays de entrada do modelo NeuMF."""
    customers = np.asarray(customers)
    products = np.asarray(products)
    return [
        customers.reshape(-1, 1),
        products.reshape(-1, 1),
        catalog[1][products].reshape(-1, 1),
        catalog[2][products].reshape(-1, 1),
    ]


def categories_for_products(products: np.ndarray, catalog: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mapeia um vetor de IDs de produto para as categorias/subcategorias do catálogo."""
    _, category_ids, subcategory_ids = catalog
    return category_ids[products], subcategory_ids[products]


# ---------------------------------------------------------------------------
# Pontuação e avaliação de ranking
# ---------------------------------------------------------------------------
def score_catalog(
    model: tf.keras.Model,
    customers: Sequence[int],
    catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 8192,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Pontua o catálogo COMPLETO para cada cliente.

    Returns:
        Matriz ``(len(customers), num_products)`` de probabilidades de interação.
    """
    product_ids, category_ids, subcategory_ids = catalog
    num_products = len(product_ids)
    customers = np.asarray(list(customers), dtype="int64")

    customers_tile = np.repeat(customers, num_products)
    products_tile = np.tile(product_ids, len(customers))
    categories_tile = np.tile(category_ids, len(customers))
    subcategories_tile = np.tile(subcategory_ids, len(customers))

    scores = model.predict(
        [
            customers_tile.reshape(-1, 1),
            products_tile.reshape(-1, 1),
            categories_tile.reshape(-1, 1),
            subcategories_tile.reshape(-1, 1),
        ],
        batch_size=batch_size,
        verbose=0,
    ).ravel()
    return scores.reshape(len(customers), num_products)


def _rank_top_k(scores: np.ndarray, excluded: Iterable[int], k: int) -> list[int]:
    """Retorna os índices dos top-k itens, ignorando os itens excluídos (já vistos)."""
    scores = scores.copy()
    excluded = list(excluded)
    if excluded:
        scores[excluded] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    candidatos = np.argpartition(-scores, k - 1)[:k]
    return candidatos[np.argsort(-scores[candidatos])].tolist()


def _dcg(relevancias: Sequence[float]) -> float:
    return float(sum(rel / math.log2(pos + 2) for pos, rel in enumerate(relevancias)))


def _zscore(scores: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Padroniza (média 0, desvio 1); eixo ``None`` padroniza o vetor inteiro."""
    scores = np.asarray(scores, dtype="float64")
    keepdims = axis is not None
    mu = scores.mean(axis=axis, keepdims=keepdims)
    sd = scores.std(axis=axis, keepdims=keepdims)
    return (scores - mu) / np.where(sd == 0, 1.0, sd)


def combine_scores(
    model_scores: np.ndarray, popularity: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Combina o prior de popularidade com o sinal personalizado do NeuMF.

        score = z(log1p(popularidade)) + alpha * z(NeuMF)

    Os dois sinais entram em escala padronizada (desvio 1), de modo que ``alpha``
    tem significado direto: ``alpha=1`` dá peso igual aos dois; ``alpha=0``
    degenera para o recomendador puramente popular.
    """
    pop_z = _zscore(np.log1p(np.asarray(popularity, dtype="float64")))
    model_z = _zscore(np.asarray(model_scores, dtype="float64"), axis=1)
    return pop_z[None, :] + alpha * model_z


def _ranking_metrics_from_scores(
    scores: np.ndarray,
    seen_by_customer: dict[int, set],
    relevant_by_customer: dict[int, set],
    customers: Sequence[int],
    k: int,
) -> dict[str, float]:
    """Calcula NDCG@k, MAP@k, Hit-Rate@k, Precision@k, Recall@k e F1@k a partir dos scores."""
    ndcg, average_precision, hit_rate = [], [], []
    precision, recall = [], []

    for row, customer in enumerate(customers):
        relevant = relevant_by_customer[customer]
        seen = seen_by_customer.get(customer, set())
        recommended = _rank_top_k(scores[row], seen, k)

        hits = len(set(recommended) & relevant)
        precision.append(hits / k if k else 0.0)
        recall.append(hits / len(relevant) if relevant else 0.0)
        hit_rate.append(1.0 if hits > 0 else 0.0)

        gains = [1.0 if item in relevant else 0.0 for item in recommended]
        ideal = [1.0] * min(len(relevant), k)
        ndcg.append(_dcg(gains) / _dcg(ideal) if _dcg(ideal) > 0 else 0.0)

        acumulado, acertos = 0.0, 0
        for pos, item in enumerate(recommended):
            if item in relevant:
                acertos += 1
                acumulado += acertos / (pos + 1)
        average_precision.append(acumulado / min(len(relevant), k) if relevant else 0.0)

    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0
    )
    return {
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"map@{k}": float(np.mean(average_precision)),
        f"hit_rate@{k}": float(np.mean(hit_rate)),
        f"precision@{k}": float(precision.mean()),
        f"recall@{k}": float(recall.mean()),
        f"f1@{k}": float(f1.mean()),
    }


def _eval_context(train_pairs: pd.DataFrame, eval_pairs: pd.DataFrame):
    """Retorna (clientes, vistos no treino, relevantes) usados na avaliação."""
    seen_by_customer = _seen_products_by_customer(train_pairs)
    relevant_by_customer = eval_pairs.groupby("c")["p"].apply(set).to_dict()
    customers = sorted(relevant_by_customer.keys())
    return customers, seen_by_customer, relevant_by_customer


def evaluate_ranking(
    model: tf.keras.Model,
    test_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
    k: int = TOP_K,
    max_customers: int | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """
    Avalia o NeuMF puro no protocolo leave-last-out.

    Para cada cliente: ranqueia o catálogo completo, **exclui os itens já vistos**
    (do treino) e compara o top-k com o item relevante (held-out).

    Returns:
        Dicionário com NDCG@k, MAP@k, Hit-Rate@k, Precision@k, Recall@k e F1@k.
    """
    customers, seen_by_customer, relevant_by_customer = _eval_context(train_pairs, test_pairs)
    if max_customers is not None:
        customers = customers[:max_customers]

    scores = score_catalog(model, customers, catalog, seed=seed)
    metricas = _ranking_metrics_from_scores(
        scores, seen_by_customer, relevant_by_customer, customers, k
    )
    metricas["num_customers"] = float(len(customers))
    metricas["num_products"] = float(scores.shape[1])
    return metricas


def evaluate_hybrid(
    model: tf.keras.Model,
    eval_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
    popularity: np.ndarray,
    alpha: float,
    k: int = TOP_K,
) -> dict[str, float]:
    """
    Avalia o recomendador **híbrido popularity-first** no protocolo leave-last-out.

    ``eval_pairs`` pode ser a validação (para selecionar ``alpha``) ou o teste
    (métrica final). Os itens já vistos (do treino) são excluídos do ranking.
    """
    customers, seen_by_customer, relevant_by_customer = _eval_context(train_pairs, eval_pairs)
    scores = combine_scores(score_catalog(model, customers, catalog), popularity, alpha)
    metricas = _ranking_metrics_from_scores(
        scores, seen_by_customer, relevant_by_customer, customers, k
    )
    metricas["num_customers"] = float(len(customers))
    metricas["num_products"] = float(scores.shape[1])
    return metricas


def select_alpha(
    model: tf.keras.Model,
    val_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
    popularity: np.ndarray,
    alphas: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
    k: int = TOP_K,
    min_gain: float = 1e-4,
) -> tuple[float, dict[float, float]]:
    """
    Escolhe o peso ``alpha`` do sinal personalizado na **validação**.

    Política conservadora (evita sobreajustar a uma validação ruidosa): só adota
    ``alpha > 0`` se ele superar o puramente popular (``alpha=0``) por pelo menos
    ``min_gain`` em NDCG@k. Caso contrário, mantém o prior de popularidade puro.

    Returns:
        (alpha_escolhido, {alpha: ndcg@k de validação}).
    """
    customers, seen_by_customer, relevant_by_customer = _eval_context(train_pairs, val_pairs)
    model_z = _zscore(score_catalog(model, customers, catalog), axis=1)
    pop_z = _zscore(np.log1p(np.asarray(popularity, dtype="float64")))

    historico: dict[float, float] = {}
    for alpha in alphas:
        scores = pop_z[None, :] + float(alpha) * model_z
        historico[float(alpha)] = _ranking_metrics_from_scores(
            scores, seen_by_customer, relevant_by_customer, customers, k
        )[f"ndcg@{k}"]

    base = historico[0.0]
    melhor_alpha = max(historico, key=historico.get)
    if historico[melhor_alpha] - base < min_gain:
        return 0.0, historico
    return melhor_alpha, historico


class RankingCheckpoint(tf.keras.callbacks.Callback):
    """
    Early stopping e restauração de pesos monitorando o **NDCG@k de validação**.

    Monitorar a métrica-alvo (ranking) em vez de um proxy como a AUC evita
    selecionar um modelo que separa bem positivos de negativos mas ordena mal o
    catálogo. Ao final do treino, restaura os pesos da melhor época.
    """

    def __init__(
        self,
        val_pairs: pd.DataFrame,
        train_pairs: pd.DataFrame,
        catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
        k: int = TOP_K,
        patience: int = 5,
        min_delta: float = 1e-5,
        verbose: int = 1,
    ):
        super().__init__()
        self.val_pairs = val_pairs
        self.train_pairs = train_pairs
        self.catalog = catalog
        self.k = k
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = -np.inf
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.history: dict[int, float] = {}

    def on_epoch_end(self, epoch, logs=None):
        metricas = evaluate_ranking(
            self.model, self.val_pairs, self.train_pairs, self.catalog, k=self.k
        )
        ndcg = metricas[f"ndcg@{self.k}"]
        self.history[epoch + 1] = ndcg
        if logs is not None:
            logs[f"val_ndcg@{self.k}"] = ndcg
        if ndcg > self.best + self.min_delta:
            self.best, self.best_epoch, self.wait = ndcg, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
        if self.verbose:
            print(
                f"    [ranking] época {epoch + 1}: val_ndcg@{self.k}={ndcg:.5f} "
                f"(melhor={self.best:.5f} na época {self.best_epoch})"
            )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def evaluate_most_popular(
    test_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    num_products: int,
    k: int = TOP_K,
) -> dict[str, float]:
    """
    Baseline não personalizado: recomenda os itens mais populares no treino.

    Usa o mesmo ranking determinístico do híbrido com ``alpha=0`` (ordena pela
    contagem de interações; empates são desfeitos pelo índice do produto). Assim,
    a comparação com o modelo é estritamente justa: ``alpha=0`` reproduz este
    baseline exatamente.

    Returns:
        O mesmo conjunto completo de métricas de ``evaluate_ranking`` (NDCG@k,
        MAP@k, Hit-Rate@k, Precision@k, Recall@k e F1@k), para que o baseline possa
        ser comparado linha a linha com o modelo.
    """
    popularity = compute_popularity_prior(train_pairs, num_products)
    customers, seen_by_customer, relevant_by_customer = _eval_context(train_pairs, test_pairs)
    scores = np.tile(_zscore(np.log1p(popularity)), (len(customers), 1))
    metricas = _ranking_metrics_from_scores(
        scores, seen_by_customer, relevant_by_customer, customers, k
    )
    metricas["num_customers"] = float(len(customers))
    metricas["num_products"] = float(scores.shape[1])
    return metricas


def evaluate_random_baseline(
    test_pairs: pd.DataFrame, train_pairs: pd.DataFrame, num_products: int, k: int = TOP_K
) -> dict[str, float]:
    """
    Esperança analítica do recomendador aleatório (candidatos = itens não vistos).
    """
    seen_by_customer = _seen_products_by_customer(train_pairs)
    relevant_by_customer = test_pairs.groupby("c")["p"].apply(set).to_dict()

    hit_rate, precision, recall = [], [], []
    for customer, relevant in relevant_by_customer.items():
        n_candidatos = num_products - len(seen_by_customer.get(customer, set()))
        if n_candidatos <= 0:
            continue
        n_relevantes = len(relevant)
        p_hit = 1.0 - math.comb(n_candidatos - n_relevantes, k) / math.comb(n_candidatos, k) if n_candidatos >= k else 1.0
        esperado_hits = k * n_relevantes / n_candidatos
        hit_rate.append(p_hit)
        precision.append(esperado_hits / k if k else 0.0)
        recall.append(esperado_hits / n_relevantes if n_relevantes else 0.0)

    return {
        f"hit_rate@{k}": float(np.mean(hit_rate)),
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
    }


def recommend_products(
    model: tf.keras.Model,
    customer_name: str,
    df: pd.DataFrame,
    customer_encoder,
    product_encoder,
    catalog: tuple[np.ndarray, np.ndarray, np.ndarray],
    top_k: int = TOP_K,
    popularity: np.ndarray | None = None,
    alpha: float = 0.0,
    product_col: str = "Product ID",
    product_name_col: str = "Product Name",
    category_col: str = "Category",
    subcategory_col: str = "Sub-Category",
) -> pd.DataFrame:
    """
    Gera o top-k personalizado para um cliente, excluindo o que ele já comprou.

    Quando ``popularity`` é informado, usa o score **híbrido** (prior de
    popularidade + ``alpha`` * sinal do NeuMF); caso contrário, usa apenas o
    NeuMF. Clientes desconhecidos (cold start) caem no baseline Most-Popular.
    """
    catalogo_produtos = df[[product_col, product_name_col, category_col, subcategory_col]].drop_duplicates(
        subset=[product_col]
    )
    catalogo_produtos = catalogo_produtos.reset_index(drop=True)

    try:
        customer_id = int(customer_encoder.transform([customer_name])[0])
    except ValueError:
        popularidade = df[product_col].value_counts().head(top_k)
        recomendados = catalogo_produtos[catalogo_produtos[product_col].isin(popularidade.index)].copy()
        recomendados["score"] = recomendados[product_col].map(popularidade).astype(float)
        recomendados = recomendados.sort_values("score", ascending=False).head(top_k)
        recomendados.insert(0, "ranking", range(1, len(recomendados) + 1))
        return recomendados.reset_index(drop=True)

    scores = score_catalog(model, [customer_id], catalog)[0]
    if popularity is not None:
        scores = combine_scores(scores[None, :], popularity, alpha)[0]
    comprados = set(product_encoder.transform(df[df["Customer Name"] == customer_name][product_col]))
    top_indices = _rank_top_k(scores, comprados, top_k)

    # Mapeia índice codificado -> Product ID -> metadados.
    product_ids = product_encoder.inverse_transform(top_indices)
    recomendados = catalogo_produtos.set_index(product_col).loc[product_ids].reset_index()
    recomendados["score"] = scores[top_indices]
    recomendados = recomendados[[product_col, product_name_col, category_col, subcategory_col, "score"]]
    recomendados.insert(0, "ranking", range(1, len(recomendados) + 1))
    return recomendados.reset_index(drop=True)
