#!/usr/bin/env python3
"""
Testes unitários para model_utils.py
"""

import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.model_utils import (
    create_model_with_regularization,
    calculate_recommendation_metrics
)

class TestModelUtils(unittest.TestCase):
    
    def setUp(self):
        """Configuração inicial para os testes"""
        # Dados de exemplo
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Criar dados sintéticos
        self.num_customers = 100
        self.num_products = 50
        self.num_categories = 5
        self.num_subcategories = 10
        
        # Parâmetros do modelo
        self.embedding_dim = 32
        self.l2_strength = 0.01
        self.dropout_rate = 0.3
        
    def test_create_model_with_regularization(self):
        """Testa a criação do modelo com regularização"""
        model = create_model_with_regularization(
            num_customers=self.num_customers,
            num_products=self.num_products,
            num_categories=self.num_categories,
            num_subcategories=self.num_subcategories,
            embedding_dim=self.embedding_dim,
            l2_strength=self.l2_strength,
            dropout_rate=self.dropout_rate
        )
        
        # Verificar se o modelo foi criado
        self.assertIsNotNone(model)
        self.assertIsInstance(model, tf.keras.Model)
        
        # Verificar se o modelo tem as camadas esperadas
        self.assertTrue(len(model.layers) > 0)
        
        # Verificar se o modelo pode fazer predições
        dummy_input = [
            np.random.randint(0, self.num_customers, (10, 1)),
            np.random.randint(0, self.num_products, (10, 1)),
            np.random.randint(0, self.num_categories, (10, 1)),
            np.random.randint(0, self.num_subcategories, (10, 1))
        ]
        
        predictions = model.predict(dummy_input, verbose=0)
        self.assertEqual(predictions.shape, (10, 1))
        
    def test_calculate_recommendation_metrics(self):
        """Testa o cálculo de métricas de recomendação"""
        # Dados de exemplo
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
        
        # Calcular métricas
        precision_5, recall_5, f1_5 = calculate_recommendation_metrics(y_true, y_pred, k=5)
        
        # Verificar se as métricas estão no intervalo correto
        self.assertGreaterEqual(precision_5, 0.0)
        self.assertLessEqual(precision_5, 1.0)
        self.assertGreaterEqual(recall_5, 0.0)
        self.assertLessEqual(recall_5, 1.0)
        self.assertGreaterEqual(f1_5, 0.0)
        self.assertLessEqual(f1_5, 1.0)
        
    def test_model_training_basic(self):
        """Testa treinamento básico do modelo"""
        model = create_model_with_regularization(
            num_customers=self.num_customers,
            num_products=self.num_products,
            num_categories=self.num_categories,
            num_subcategories=self.num_subcategories,
            embedding_dim=16,  # Menor para teste rápido
            l2_strength=0.01,
            dropout_rate=0.2
        )
        
        # Dados sintéticos para treinamento
        n_samples = 200
        X_train = [
            np.random.randint(0, self.num_customers, (n_samples, 1)),
            np.random.randint(0, self.num_products, (n_samples, 1)),
            np.random.randint(0, self.num_categories, (n_samples, 1)),
            np.random.randint(0, self.num_subcategories, (n_samples, 1))
        ]
        y_train = np.random.random((n_samples, 1))
        
        # Treinar por poucas épocas
        history = model.fit(
            X_train, y_train,
            epochs=2,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        
        # Verificar se o histórico foi criado
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)
        
        # Verificar se houve pelo menos 2 épocas
        self.assertEqual(len(history.history['loss']), 2)

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