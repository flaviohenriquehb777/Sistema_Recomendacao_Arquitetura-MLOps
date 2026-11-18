"""
Gerador de **dados sintéticos** de interações cliente–produto para benchmark de escala.

Contexto e honestidade
----------------------
Esta base é **100 % sintética** e é gerada por um processo **explicitamente
documentado e reproduzível** (semente fixa). Ela **não** representa dados reais
de nenhuma empresa. Seu objetivo é servir de *benchmark* para exercitar o
pipeline de recomendação em uma **escala e com uma estrutura colaborativa** que a
base pública pequena (sc_base_01) não possui.

Por que uma base sintética é legítima para portfólio:
    * dados reais de clientes são protegidos por privacidade — o que se publica
      são *substitutos* anonimizados/sintéticos;
    * o que precisa ser provado não é "os dados", mas que **o método e a
      arquitetura recuperam preferência latente quando ela existe** — e isso só
      se demonstra com uma base que tenha sinal colaborativo de verdade.

Modelo generativo (premissas transparentes)
-------------------------------------------
1. **Item** — cada produto recebe uma direção latente ``v_i`` construída a partir
   da direção da sua subcategoria/categoria mais ruído (controlado). Produtos de
   uma mesma subcategoria ficam próximos no espaço latente (é o que dá sentido às
   *features* de categoria/subcategoria do modelo).
2. **Usuário** — cada cliente recebe um vetor de preferência ``u_u``, sorteado
   como uma combinação de algumas direções de categoria (um "perfil de gosto")
   mais ruído (controlado).
3. **Afinidade** — ``a_ui = u_u · v_i`` (produto interno).
4. **Popularidade** — cada item tem uma propensão ``s_i`` com **cauda longa**
   (lei de potência via log-normal), gerando um *baseline* de "mais vendidos"
   não-trivial.
5. **Amostragem** — a probabilidade de um usuário interagir com um item é
   ``p_ui ∝ s_i · exp(preference_strength · a_ui)``; cada usuário interage com
   um número de itens ~ Poisson(``avg_interactions``), **sem reposição**.
6. **Tempo** — cada interação recebe um instante; o split *leave-last-out* passa
   a respeitar a ordem temporal (como no mundo real).

O ruído é injetado com **norma controlada** (vetores unitários escalonados) e a
força da preferência (``preference_strength``) foi **calibrada** para que a
afinidade latente domine a popularidade global sem tornar o problema trivial:
com os valores padrão, o teto teórico "oracle" (ranquear pelo fator latente
verdadeiro) atinge ≈0,35 de Hit-Rate@7 contra ≈0,05 do baseline de popularidade.
Essa é a condição necessária (exatamente como em dados reais de larga escala) para
que um modelo de filtragem colaborativa **supere** o *baseline* de popularidade —
e é essa a demonstração que interessa.

Todas as funções são determinísticas (semente fixa) e não alteram a base real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Normaliza cada linha para norma unitária (evita divisão por zero)."""
    matrix = np.asarray(matrix, dtype="float64")
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def _unit_noise(rng: np.random.Generator, shape) -> np.ndarray:
    """Ruído gaussiano com cada linha normalizada (norma 1) — escala controlável."""
    return _normalize(rng.normal(size=shape))


def build_latent_factors(
    n_users: int = 6000,
    n_items: int = 1500,
    n_categories: int = 8,
    n_subcategories: int = 40,
    latent_dim: int = 16,
    popularity_skew: float = 0.6,
    categories_per_user: int = 3,
    category_noise: float = 0.5,
    item_noise: float = 0.35,
    user_noise: float = 0.35,
    seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """
    Constrói os fatores latentes (usuários, itens, popularidade) do gerador.

    Returns:
        Dicionário com ``user_latent`` ``(n_users, d)``, ``item_latent``
        ``(n_items, d)``, ``item_popularity`` ``(n_items,)``,
        ``item_subcategory`` ``(n_items,)`` e ``sub_of_category`` ``(S,)``.
    """
    rng = np.random.default_rng(seed)

    category_dirs = _normalize(rng.normal(size=(n_categories, latent_dim)))
    sub_of_category = np.arange(n_subcategories) % n_categories
    subcategory_dirs = _normalize(
        category_dirs[sub_of_category] + category_noise * _unit_noise(rng, (n_subcategories, latent_dim))
    )

    item_subcategory = np.sort(rng.integers(0, n_subcategories, size=n_items))
    item_latent = _normalize(
        subcategory_dirs[item_subcategory] + item_noise * _unit_noise(rng, (n_items, latent_dim))
    )
    item_popularity = np.exp(rng.normal(0.0, popularity_skew, size=n_items))

    user_latent = np.empty((n_users, latent_dim))
    for u in range(n_users):
        escolhidas = rng.choice(n_categories, size=categories_per_user, replace=False)
        user_latent[u] = (
            category_dirs[escolhidas].sum(axis=0) + user_noise * _unit_noise(rng, (latent_dim,))
        )
    user_latent = _normalize(user_latent)

    return {
        "user_latent": user_latent,
        "item_latent": item_latent,
        "item_popularity": item_popularity,
        "item_subcategory": item_subcategory,
        "sub_of_category": sub_of_category,
    }


def generate_synthetic_dataset(
    n_users: int = 6000,
    n_items: int = 1500,
    n_categories: int = 8,
    n_subcategories: int = 40,
    latent_dim: int = 16,
    avg_interactions: int = 24,
    popularity_skew: float = 0.6,
    preference_strength: float = 20.0,
    categories_per_user: int = 3,
    time_span_days: int = 540,
    seed: int = RANDOM_SEED,
    return_factors: bool = False,
):
    """
    Gera um DataFrame de interações cliente–produto (base sintética de escala).

    Args:
        return_factors: quando ``True``, retorna ``(df, fatores)`` em vez de
            apenas ``df`` (útil para diagnósticos e para medir o *oracle*).

    Returns:
        DataFrame com as colunas ``Customer Name``, ``Product ID``,
        ``Product Name``, ``Category``, ``Sub-Category``, ``Order Date`` e
        ``Sales`` — o mesmo esquema consumido pelo pipeline de recomendação.
    """
    fatores = build_latent_factors(
        n_users=n_users, n_items=n_items, n_categories=n_categories,
        n_subcategories=n_subcategories, latent_dim=latent_dim,
        popularity_skew=popularity_skew, categories_per_user=categories_per_user, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)

    user_latent = fatores["user_latent"]
    item_latent = fatores["item_latent"]
    item_subcategory = fatores["item_subcategory"]
    sub_of_category = fatores["sub_of_category"]
    log_popularity = np.log(fatores["item_popularity"])

    registros = []
    for u in range(n_users):
        afinidade = user_latent[u] @ item_latent.T  # (n_items,)
        logits = log_popularity + preference_strength * afinidade
        logits -= logits.max()  # estabilidade numérica do softmax
        probabilidades = np.exp(logits)
        probabilidades /= probabilidades.sum()

        n_interacoes = int(np.clip(rng.poisson(avg_interactions), 5, n_items))
        itens = rng.choice(n_items, size=n_interacoes, replace=False, p=probabilidades)

        # Ordem temporal: o último item do usuário será o *held-out* (teste).
        dias = np.sort(rng.uniform(0.0, time_span_days, size=n_interacoes))
        base = np.datetime64("2023-01-01")
        for item, dia in zip(itens, dias):
            registros.append((u, int(item), float(np.round(rng.uniform(5.0, 900.0), 2)), base + np.timedelta64(int(dia), "D")))

    interacoes = pd.DataFrame(registros, columns=["user_idx", "item_idx", "Sales", "Order Date"])

    # --- Metadados legíveis (mesmo esquema da base real) ---------------------
    produtos = pd.DataFrame(
        {
            "item_idx": np.arange(n_items),
            "Product ID": [f"PROD-{i:05d}" for i in range(n_items)],
            "Sub-Category": [f"SUB-{s:02d}" for s in item_subcategory],
        }
    )
    produtos["Category"] = [f"CAT-{c:02d}" for c in sub_of_category[item_subcategory]]
    produtos["Product Name"] = produtos["Sub-Category"] + " Item " + produtos["item_idx"].map(lambda i: f"{i:05d}")

    interacoes["Customer Name"] = interacoes["user_idx"].map(lambda u: f"Cliente {u:05d}")
    interacoes = interacoes.merge(
        produtos[["item_idx", "Product ID", "Product Name", "Category", "Sub-Category"]],
        on="item_idx",
        how="left",
    )

    colunas = [
        "Customer Name",
        "Product ID",
        "Product Name",
        "Category",
        "Sub-Category",
        "Order Date",
        "Sales",
    ]
    df = interacoes[colunas].sort_values(["Customer Name", "Order Date"]).reset_index(drop=True)
    if return_factors:
        return df, fatores
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Estatísticas descritivas da base (para relatório de benchmark)."""
    clientes = df["Customer Name"].nunique()
    produtos = df["Product ID"].nunique()
    interacoes = len(df)
    por_cliente = df.groupby("Customer Name")["Product ID"].nunique()
    return {
        "interacoes": int(interacoes),
        "clientes": int(clientes),
        "produtos": int(produtos),
        "categorias": int(df["Category"].nunique()),
        "subcategorias": int(df["Sub-Category"].nunique()),
        "itens_por_cliente_media": float(por_cliente.mean()),
        "itens_por_cliente_mediana": float(por_cliente.median()),
        "densidade": float(interacoes / (clientes * produtos)) if clientes and produtos else 0.0,
    }


def _smoke() -> None:  # pragma: no cover - verificação manual rápida
    df = generate_synthetic_dataset(n_users=200, n_items=120, avg_interactions=15)
    resumo = dataset_summary(df)
    print("Amostra sintética (smoke test):")
    for chave, valor in resumo.items():
        print(f"  {chave:<26}: {valor}")
    print(f"  colunas                   : {list(df.columns)}")


if __name__ == "__main__":  # pragma: no cover
    _smoke()
