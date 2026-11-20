"""
Testes unitários para model_utils.py
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.model_utils import (
    combine_scores,
    compute_popularity_prior,
    create_ranking_model,
    evaluate_most_popular,
    to_model_inputs,
    train_ranking_model,
)


class TestModelUtils(unittest.TestCase):
    """Testes do NeuMF (ranking) e do recomendador híbrido popularity-first."""

    def setUp(self):
        """Configuração inicial para os testes"""
        np.random.seed(42)
        tf.keras.utils.set_random_seed(42)

        # Dimensões do problema de teste
        self.num_customers = 100
        self.num_products = 50
        self.num_categories = 5
        self.num_subcategories = 10

    def test_create_ranking_model(self):
        """O NeuMF é criado e produz probabilidades válidas a partir das 4 entradas."""
        model = create_ranking_model(
            num_customers=self.num_customers,
            num_products=self.num_products,
            num_categories=self.num_categories,
            num_subcategories=self.num_subcategories,
            embedding_dim=16,
        )

        self.assertIsInstance(model, tf.keras.Model)
        self.assertTrue(len(model.layers) > 0)

        dummy_input = [
            np.random.randint(0, self.num_customers, (10, 1)),
            np.random.randint(0, self.num_products, (10, 1)),
            np.random.randint(0, self.num_categories, (10, 1)),
            np.random.randint(0, self.num_subcategories, (10, 1)),
        ]
        predictions = model.predict(dummy_input, verbose=0)

        self.assertEqual(predictions.shape, (10, 1))
        self.assertTrue(np.all(predictions >= 0.0) and np.all(predictions <= 1.0))

    def test_compute_popularity_prior(self):
        """O prior de popularidade conta as interações por produto no treino."""
        train_pairs = pd.DataFrame({"c": [0, 0, 1, 1, 1], "p": [0, 1, 1, 1, 2]})

        prior = compute_popularity_prior(train_pairs, num_products=4)

        np.testing.assert_array_equal(prior, np.array([1.0, 3.0, 1.0, 0.0]))

    def test_combine_scores_alpha_zero_is_popularity(self):
        """Com alpha=0 o score é apenas o prior de popularidade (sem personalização)."""
        popularity = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
        model_scores = np.random.random((3, 5))

        scores = combine_scores(model_scores, popularity, alpha=0.0)

        self.assertEqual(scores.shape, (3, 5))
        np.testing.assert_allclose(scores[0], scores[1])
        np.testing.assert_allclose(scores[1], scores[2])

        # Com alpha>0 o sinal do NeuMF passa a diferenciar as linhas.
        scores_personalizado = combine_scores(model_scores, popularity, alpha=1.0)
        self.assertFalse(np.allclose(scores_personalizado[0], scores_personalizado[1]))

    def test_to_model_inputs(self):
        """to_model_inputs devolve os 4 arrays do NeuMF com o contexto do catálogo."""
        catalog = (
            np.array([0, 1, 2], dtype="int32"),
            np.array([0, 1, 0], dtype="int32"),  # categoria por produto
            np.array([0, 1, 1], dtype="int32"),  # subcategoria por produto
        )
        customers = np.array([7, 9])
        products = np.array([0, 2])

        inputs = to_model_inputs(customers, products, catalog)

        self.assertEqual(len(inputs), 4)
        for arr in inputs:
            self.assertEqual(arr.shape, (2, 1))
        np.testing.assert_array_equal(inputs[0].ravel(), customers)
        np.testing.assert_array_equal(inputs[1].ravel(), products)
        np.testing.assert_array_equal(inputs[2].ravel(), np.array([0, 0]))
        np.testing.assert_array_equal(inputs[3].ravel(), np.array([0, 1]))

    def test_train_ranking_model_basic(self):
        """O loop de treino de ranking roda e registra o histórico por época."""
        model = create_ranking_model(
            num_customers=self.num_customers,
            num_products=self.num_products,
            num_categories=self.num_categories,
            num_subcategories=self.num_subcategories,
            embedding_dim=8,
        )

        n_samples = 200
        history = train_ranking_model(
            model,
            train_customers=np.random.randint(0, self.num_customers, n_samples),
            train_products=np.random.randint(0, self.num_products, n_samples),
            categories=np.random.randint(0, self.num_categories, n_samples),
            subcategories=np.random.randint(0, self.num_subcategories, n_samples),
            labels=np.random.randint(0, 2, n_samples).astype("float32"),
            epochs=2,
            batch_size=32,
            verbose=0,
        )

        self.assertIn("loss", history.history)
        self.assertEqual(len(history.history["loss"]), 2)

    def test_evaluate_most_popular_baseline(self):
        """O baseline popular roda no leave-last-out e devolve métricas no intervalo [0, 1]."""
        train_pairs = pd.DataFrame({"c": [0, 0, 0, 1, 1, 1], "p": [0, 1, 2, 0, 1, 2]})
        test_pairs = pd.DataFrame({"c": [0, 1], "p": [3, 3]})

        metricas = evaluate_most_popular(test_pairs, train_pairs, num_products=5, k=2)

        for chave in ["ndcg@2", "map@2", "hit_rate@2", "precision@2", "recall@2", "f1@2"]:
            self.assertIn(chave, metricas)
        for chave, valor in metricas.items():
            if "@" not in chave:
                continue
            self.assertGreaterEqual(valor, 0.0)
            self.assertLessEqual(valor, 1.0)


class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Configuração inicial para testes de processamento de dados"""
        # Criar DataFrame de exemplo
        self.df = pd.DataFrame({
            'Customer Name': ['Cliente A', 'Cliente B', 'Cliente A', 'Cliente C'],
            'Product ID': ['P001', 'P002', 'P003', 'P001'],
            'Product Name': ['Produto 1', 'Produto 2', 'Produto 3', 'Produto 1'],
            'Category': ['Cat1', 'Cat2', 'Cat1', 'Cat1'],
            'Sub-Category': ['Sub1', 'Sub2', 'Sub1', 'Sub1'],
            'Sales': [100.0, 200.0, 150.0, 120.0]
        })
        
    def test_label_encoding(self):
        """Testa a codificação de labels"""
        customer_encoder = LabelEncoder()
        product_encoder = LabelEncoder()
        
        # Codificar
        customer_encoded = customer_encoder.fit_transform(self.df['Customer Name'])
        product_encoded = product_encoder.fit_transform(self.df['Product ID'])
        
        # Verificar se a codificação funcionou
        self.assertEqual(len(customer_encoded), len(self.df))
        self.assertEqual(len(product_encoded), len(self.df))
        
        # Verificar se os valores estão no intervalo correto
        self.assertTrue(all(0 <= x < len(customer_encoder.classes_) for x in customer_encoded))
        self.assertTrue(all(0 <= x < len(product_encoder.classes_) for x in product_encoded))
        
    def test_sales_normalization(self):
        """Testa a normalização das vendas"""
        scaler = MinMaxScaler()
        sales_normalized = scaler.fit_transform(self.df[['Sales']])
        
        # Verificar se a normalização funcionou
        self.assertEqual(len(sales_normalized), len(self.df))
        self.assertTrue(all(0 <= x <= 1 for x in sales_normalized.flatten()))
        
        # Verificar se min e max estão corretos
        self.assertAlmostEqual(sales_normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(sales_normalized.max(), 1.0, places=5)

if __name__ == '__main__':
    # Configurar TensorFlow para não usar GPU nos testes
    tf.config.set_visible_devices([], 'GPU')
    
    # Executar testes
    unittest.main(verbosity=2)
