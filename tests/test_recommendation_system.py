#!/usr/bin/env python3
"""
Testes para o sistema de recomendação
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestRecommendationSystem(unittest.TestCase):
    
    def setUp(self):
        """Configuração inicial para os testes"""
        # Criar dados de exemplo
        self.sample_data = pd.DataFrame({
            'Customer Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
            'Product ID': ['P001', 'P002', 'P003', 'P001', 'P004'],
            'Product Name': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Monitor'],
            'Category': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology'],
            'Sub-Category': ['Computers', 'Accessories', 'Accessories', 'Computers', 'Accessories'],
            'Sales': [1000.0, 25.0, 75.0, 1200.0, 300.0]
        })
        
    def test_data_validation(self):
        """Testa validação dos dados de entrada"""
        # Verificar se os dados têm as colunas necessárias
        required_columns = ['Customer Name', 'Product ID', 'Product Name', 'Category', 'Sub-Category', 'Sales']
        
        for col in required_columns:
            self.assertIn(col, self.sample_data.columns)
            
        # Verificar se não há valores nulos nas colunas críticas
        critical_columns = ['Customer Name', 'Product ID', 'Sales']
        for col in critical_columns:
            self.assertFalse(self.sample_data[col].isnull().any())
            
        # Verificar se Sales são numéricas e positivas
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['Sales']))
        self.assertTrue((self.sample_data['Sales'] >= 0).all())
        
    def test_unique_counts(self):
        """Testa contagem de elementos únicos"""
        unique_customers = self.sample_data['Customer Name'].nunique()
        unique_products = self.sample_data['Product ID'].nunique()
        unique_categories = self.sample_data['Category'].nunique()
        
        self.assertEqual(unique_customers, 3)  # Alice, Bob, Charlie
        self.assertEqual(unique_products, 4)   # P001, P002, P003, P004
        self.assertEqual(unique_categories, 1) # Technology
        
    def test_customer_product_matrix(self):
        """Testa criação da matriz cliente-produto"""
        # Criar matriz pivot
        customer_product_matrix = self.sample_data.pivot_table(
            index='Customer Name',
            columns='Product ID',
            values='Sales',
            fill_value=0
        )
        
        # Verificar dimensões
        self.assertEqual(customer_product_matrix.shape[0], 3)  # 3 clientes
        self.assertEqual(customer_product_matrix.shape[1], 4)  # 4 produtos
        
        # Verificar se Alice comprou P001 e P003
        self.assertGreater(customer_product_matrix.loc['Alice', 'P001'], 0)
        self.assertGreater(customer_product_matrix.loc['Alice', 'P003'], 0)
        self.assertEqual(customer_product_matrix.loc['Alice', 'P002'], 0)
        
    def test_sales_statistics(self):
        """Testa estatísticas das vendas"""
        sales_stats = self.sample_data['Sales'].describe()
        
        # Verificar estatísticas básicas
        self.assertEqual(sales_stats['count'], 5)
        self.assertEqual(sales_stats['min'], 25.0)
        self.assertEqual(sales_stats['max'], 1200.0)
        self.assertAlmostEqual(sales_stats['mean'], 520.0, places=1)
        
    def test_recommendation_logic_basic(self):
        """Testa lógica básica de recomendação"""
        # Simular função de recomendação simples baseada em popularidade
        def recommend_popular_products(df, customer_name, n_recommendations=3):
            # Produtos que o cliente ainda não comprou
            customer_products = df[df['Customer Name'] == customer_name]['Product ID'].unique()
            available_products = df[~df['Product ID'].isin(customer_products)]
            
            # Recomendar produtos mais vendidos
            popular_products = available_products.groupby('Product ID')['Sales'].sum().nlargest(n_recommendations)
            
            return popular_products.index.tolist()
        
        # Testar recomendações para Alice
        recommendations = recommend_popular_products(self.sample_data, 'Alice', 2)
        
        # Alice já comprou P001 e P003, então deve recomendar P004 e P002
        self.assertIn('P004', recommendations)  # Monitor (300.0)
        self.assertIn('P002', recommendations)  # Mouse (25.0)
        self.assertNotIn('P001', recommendations)  # Já comprou
        self.assertNotIn('P003', recommendations)  # Já comprou
        
    def test_category_distribution(self):
        """Testa distribuição por categoria"""
        category_sales = self.sample_data.groupby('Category')['Sales'].sum()
        
        # Verificar se Technology é a única categoria
        self.assertEqual(len(category_sales), 1)
        self.assertIn('Technology', category_sales.index)
        self.assertEqual(category_sales['Technology'], 2600.0)  # Soma de todas as vendas
        
    def test_subcategory_distribution(self):
        """Testa distribuição por sub-categoria"""
        subcategory_sales = self.sample_data.groupby('Sub-Category')['Sales'].sum()
        
        # Verificar sub-categorias
        self.assertIn('Computers', subcategory_sales.index)
        self.assertIn('Accessories', subcategory_sales.index)
        
        # Computers: P001 (1000 + 1200) = 2200
        # Accessories: P002 (25) + P003 (75) + P004 (300) = 400
        self.assertEqual(subcategory_sales['Computers'], 2200.0)
        self.assertEqual(subcategory_sales['Accessories'], 400.0)

class TestModelIntegration(unittest.TestCase):
    
    def test_model_file_structure(self):
        """Testa se a estrutura de arquivos do modelo está correta"""
        # Verificar se as pastas necessárias existem
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        expected_dirs = [
            'src',
            'src/config',
            'src/scripts',
            'models',
            'notebooks',
            'tests'
        ]
        
        for dir_path in expected_dirs:
            full_path = os.path.join(project_root, dir_path)
            self.assertTrue(os.path.exists(full_path), f"Diretório {dir_path} não encontrado")
            
    def test_config_files_exist(self):
        """Testa se os arquivos de configuração existem"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        expected_files = [
            'src/config/paths.py',
            'src/config/model_utils.py',
            'src/config/dagshub_config.py',
            'requirements.txt',
            'README.md'
        ]
        
        for file_path in expected_files:
            full_path = os.path.join(project_root, file_path)
            self.assertTrue(os.path.exists(full_path), f"Arquivo {file_path} não encontrado")

class TestPerformanceMetrics(unittest.TestCase):
    
    def test_precision_at_k(self):
        """Testa cálculo de precision@k"""
        def precision_at_k(y_true, y_pred, k):
            # Ordenar predições em ordem decrescente
            sorted_indices = np.argsort(y_pred)[::-1]
            top_k_indices = sorted_indices[:k]
            
            # Calcular precision
            relevant_items = np.sum(y_true[top_k_indices])
            return relevant_items / k if k > 0 else 0
        
        # Dados de teste
        y_true = np.array([1, 0, 1, 0, 1])  # 3 itens relevantes
        y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7])  # Predições
        
        # Testar precision@3
        precision_3 = precision_at_k(y_true, y_pred, 3)
        
        # Top 3 predições: índices 0, 2, 4 (scores 0.9, 0.8, 0.7)
        # y_true[0] = 1, y_true[2] = 1, y_true[4] = 1
        # Precision@3 = 3/3 = 1.0
        self.assertAlmostEqual(precision_3, 1.0, places=2)
        
    def test_recall_at_k(self):
        """Testa cálculo de recall@k"""
        def recall_at_k(y_true, y_pred, k):
            total_relevant = np.sum(y_true)
            if total_relevant == 0:
                return 0
                
            sorted_indices = np.argsort(y_pred)[::-1]
            top_k_indices = sorted_indices[:k]
            relevant_items = np.sum(y_true[top_k_indices])
            
            return relevant_items / total_relevant
        
        # Dados de teste
        y_true = np.array([1, 0, 1, 0, 1])  # 3 itens relevantes
        y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7])
        
        # Testar recall@3
        recall_3 = recall_at_k(y_true, y_pred, 3)
        
        # 3 itens relevantes encontrados de 3 totais = 3/3 = 1.0
        self.assertAlmostEqual(recall_3, 1.0, places=2)

if __name__ == '__main__':
    unittest.main(verbosity=2)