#!/usr/bin/env python3
"""
Script para criar modelo ensemble combinando diferentes arquiteturas de recomendação.
Implementa múltiplas abordagens e combina suas predições para maximizar a performance.
"""

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

# Importar utilitários personalizados
from src.config.paths import DADOS_BRUTOS, DADOS_TRATADOS
from src.config.auxiliares_ml import downcast_dataframe

def create_deep_model(num_customers, num_products, num_categories, num_subcategories, 
                     embedding_dim=64, dropout_rate=0.3):
    """Criar modelo deep learning com arquitetura mais profunda."""
    
    # Inputs
    customer_input = tf.keras.layers.Input(shape=(1,), name='customer_input')
    product_input = tf.keras.layers.Input(shape=(1,), name='product_input')
    category_input = tf.keras.layers.Input(shape=(1,), name='category_input')
    subcategory_input = tf.keras.layers.Input(shape=(1,), name='subcategory_input')
    
    # Embeddings
    customer_embedding = tf.keras.layers.Embedding(
        num_customers, embedding_dim, name='customer_embedding'
    )(customer_input)
    
    product_embedding = tf.keras.layers.Embedding(
        num_products, embedding_dim, name='product_embedding'
    )(product_input)
    
    category_embedding = tf.keras.layers.Embedding(
        num_categories, embedding_dim//2, name='category_embedding'
    )(category_input)
    
    subcategory_embedding = tf.keras.layers.Embedding(
        num_subcategories, embedding_dim//2, name='subcategory_embedding'
    )(subcategory_input)
    
    # Flatten embeddings
    customer_flat = tf.keras.layers.Flatten()(customer_embedding)
    product_flat = tf.keras.layers.Flatten()(product_embedding)
    category_flat = tf.keras.layers.Flatten()(category_embedding)
    subcategory_flat = tf.keras.layers.Flatten()(subcategory_embedding)
    
    # Concatenar embeddings
    concat = tf.keras.layers.Concatenate()([
        customer_flat, product_flat, category_flat, subcategory_flat
    ])
    
    # Arquitetura profunda
    dense1 = tf.keras.layers.Dense(256, activation='relu')(concat)
    dropout1 = tf.keras.layers.Dropout(dropout_rate)(dense1)
    
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(dropout_rate)(dense2)
    
    dense3 = tf.keras.layers.Dense(64, activation='relu')(dropout2)
    dropout3 = tf.keras.layers.Dropout(dropout_rate)(dense3)
    
    dense4 = tf.keras.layers.Dense(32, activation='relu')(dropout3)
    
    # Output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense4)
    
    # Criar modelo
    model = tf.keras.Model(
        inputs=[customer_input, product_input, category_input, subcategory_input],
        outputs=output
    )
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_wide_model(num_customers, num_products, num_categories, num_subcategories, 
                     embedding_dim=32):
    """Criar modelo wide com embeddings menores mas mais camadas paralelas."""
    
    # Inputs
    customer_input = tf.keras.layers.Input(shape=(1,), name='customer_input')
    product_input = tf.keras.layers.Input(shape=(1,), name='product_input')
    category_input = tf.keras.layers.Input(shape=(1,), name='category_input')
    subcategory_input = tf.keras.layers.Input(shape=(1,), name='subcategory_input')
    
    # Embeddings menores
    customer_embedding = tf.keras.layers.Embedding(
        num_customers, embedding_dim, name='customer_embedding'
    )(customer_input)
    
    product_embedding = tf.keras.layers.Embedding(
        num_products, embedding_dim, name='product_embedding'
    )(product_input)
    
    category_embedding = tf.keras.layers.Embedding(
        num_categories, embedding_dim//2, name='category_embedding'
    )(category_input)
    
    subcategory_embedding = tf.keras.layers.Embedding(
        num_subcategories, embedding_dim//2, name='subcategory_embedding'
    )(subcategory_input)
    
    # Flatten embeddings
    customer_flat = tf.keras.layers.Flatten()(customer_embedding)
    product_flat = tf.keras.layers.Flatten()(product_embedding)
    category_flat = tf.keras.layers.Flatten()(category_embedding)
    subcategory_flat = tf.keras.layers.Flatten()(subcategory_embedding)
    
    # Múltiplas branches paralelas
    # Branch 1: Customer-Product interaction
    cp_concat = tf.keras.layers.Concatenate()([customer_flat, product_flat])
    cp_dense = tf.keras.layers.Dense(64, activation='relu')(cp_concat)
    
    # Branch 2: Category-Subcategory interaction
    cs_concat = tf.keras.layers.Concatenate()([category_flat, subcategory_flat])
    cs_dense = tf.keras.layers.Dense(32, activation='relu')(cs_concat)
    
    # Branch 3: All features
    all_concat = tf.keras.layers.Concatenate()([customer_flat, product_flat, category_flat, subcategory_flat])
    all_dense = tf.keras.layers.Dense(96, activation='relu')(all_concat)
    
    # Combinar branches
    final_concat = tf.keras.layers.Concatenate()([cp_dense, cs_dense, all_dense])
    final_dense = tf.keras.layers.Dense(64, activation='relu')(final_concat)
    
    # Output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(final_dense)
    
    # Criar modelo
    model = tf.keras.Model(
        inputs=[customer_input, product_input, category_input, subcategory_input],
        outputs=output
    )
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_attention_model(num_customers, num_products, num_categories, num_subcategories, 
                          embedding_dim=48):
    """Criar modelo com mecanismo de atenção simplificado."""
    
    # Inputs
    customer_input = tf.keras.layers.Input(shape=(1,), name='customer_input')
    product_input = tf.keras.layers.Input(shape=(1,), name='product_input')
    category_input = tf.keras.layers.Input(shape=(1,), name='category_input')
    subcategory_input = tf.keras.layers.Input(shape=(1,), name='subcategory_input')
    
    # Embeddings
    customer_embedding = tf.keras.layers.Embedding(
        num_customers, embedding_dim, name='customer_embedding'
    )(customer_input)
    
    product_embedding = tf.keras.layers.Embedding(
        num_products, embedding_dim, name='product_embedding'
    )(product_input)
    
    category_embedding = tf.keras.layers.Embedding(
        num_categories, embedding_dim//2, name='category_embedding'
    )(category_input)
    
    subcategory_embedding = tf.keras.layers.Embedding(
        num_subcategories, embedding_dim//2, name='subcategory_embedding'
    )(subcategory_input)
    
    # Flatten embeddings
    customer_flat = tf.keras.layers.Flatten()(customer_embedding)
    product_flat = tf.keras.layers.Flatten()(product_embedding)
    category_flat = tf.keras.layers.Flatten()(category_embedding)
    subcategory_flat = tf.keras.layers.Flatten()(subcategory_embedding)
    
    # Concatenar embeddings diretamente (versão simplificada)
    concat = tf.keras.layers.Concatenate()([
        customer_flat, product_flat, category_flat, subcategory_flat
    ])
    
    # Mecanismo de atenção simplificado usando Dense layers
    attention_dense = tf.keras.layers.Dense(128, activation='relu')(concat)
    attention_weights = tf.keras.layers.Dense(concat.shape[-1], activation='softmax')(attention_dense)
    
    # Aplicar atenção através de multiplicação elemento a elemento
    weighted_features = tf.keras.layers.Multiply()([concat, attention_weights])
    
    # Reduzir dimensionalidade
    weighted_embeddings = tf.keras.layers.Dense(96, activation='relu')(weighted_features)
    
    # Camadas densas
    dense1 = tf.keras.layers.Dense(128, activation='relu')(weighted_embeddings)
    dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
    
    # Output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
    
    # Criar modelo
    model = tf.keras.Model(
        inputs=[customer_input, product_input, category_input, subcategory_input],
        outputs=output
    )
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss='mse',
        metrics=['mae']
    )
    
    return model

class EnsembleRecommendationModel:
    """Classe para modelo ensemble de recomendação."""
    
    def __init__(self):
        self.models = {}
        self.ml_models = {}
        self.weights = {}
        self.is_trained = False
        
    def add_deep_model(self, num_customers, num_products, num_categories, num_subcategories):
        """Adicionar modelo deep learning profundo."""
        self.models['deep'] = create_deep_model(
            num_customers, num_products, num_categories, num_subcategories
        )
        
    def add_wide_model(self, num_customers, num_products, num_categories, num_subcategories):
        """Adicionar modelo wide."""
        self.models['wide'] = create_wide_model(
            num_customers, num_products, num_categories, num_subcategories
        )
        
    def add_attention_model(self, num_customers, num_products, num_categories, num_subcategories):
        """Adicionar modelo com atenção."""
        self.models['attention'] = create_attention_model(
            num_customers, num_products, num_categories, num_subcategories
        )
        
    def add_ml_models(self, X_features):
        """Adicionar modelos de machine learning tradicionais."""
        self.ml_models['random_forest'] = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.ml_models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, random_state=42
        )
        self.ml_models['ridge'] = Ridge(alpha=1.0, random_state=42)
        
    def train(self, X_dl, X_ml, y, validation_data=None, epochs=20):
        """Treinar todos os modelos do ensemble."""
        
        print("Treinando modelos deep learning...")
        
        # Treinar modelos deep learning
        for name, model in self.models.items():
            print(f"Treinando modelo {name}...")
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            if validation_data:
                X_val_dl, X_val_ml, y_val = validation_data
                validation_data_formatted = (X_val_dl, y_val)
            else:
                validation_data_formatted = None
                
            model.fit(
                X_dl, y,
                epochs=epochs,
                batch_size=32,
                validation_data=validation_data_formatted,
                callbacks=callbacks,
                verbose=1
            )
            
        print("Treinando modelos de machine learning...")
        
        # Treinar modelos ML tradicionais
        for name, model in self.ml_models.items():
            print(f"Treinando modelo {name}...")
            model.fit(X_ml, y)
            
        # Calcular pesos baseados na performance de validação
        self._calculate_weights(X_dl, X_ml, y, validation_data)
        
        self.is_trained = True
        print("Treinamento do ensemble concluído!")
        
    def _calculate_weights(self, X_dl, X_ml, y, validation_data=None):
        """Calcular pesos para cada modelo baseado na performance."""
        
        if validation_data:
            X_val_dl, X_val_ml, y_val = validation_data
        else:
            # Usar dados de treino se não houver validação
            X_val_dl, X_val_ml, y_val = X_dl, X_ml, y
            
        model_scores = {}
        
        # Avaliar modelos deep learning
        for name, model in self.models.items():
            pred = model.predict(X_val_dl, verbose=0).flatten()
            mse = mean_squared_error(y_val, pred)
            model_scores[name] = 1 / (1 + mse)  # Inverso do MSE
            
        # Avaliar modelos ML
        for name, model in self.ml_models.items():
            pred = model.predict(X_val_ml)
            mse = mean_squared_error(y_val, pred)
            model_scores[name] = 1 / (1 + mse)
            
        # Normalizar pesos
        total_score = sum(model_scores.values())
        self.weights = {name: score/total_score for name, score in model_scores.items()}
        
        print("Pesos calculados:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.4f}")
            
    def predict(self, X_dl, X_ml):
        """Fazer predições usando ensemble."""
        
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
            
        predictions = []
        
        # Predições dos modelos deep learning
        for name, model in self.models.items():
            pred = model.predict(X_dl, verbose=0).flatten()
            weighted_pred = pred * self.weights[name]
            predictions.append(weighted_pred)
            
        # Predições dos modelos ML
        for name, model in self.ml_models.items():
            pred = model.predict(X_ml)
            weighted_pred = pred * self.weights[name]
            predictions.append(weighted_pred)
            
        # Combinar predições
        ensemble_prediction = np.sum(predictions, axis=0)
        
        return ensemble_prediction
        
    def save(self, base_path="models/ensemble"):
        """Salvar todos os modelos do ensemble."""
        
        os.makedirs(base_path, exist_ok=True)
        
        # Salvar modelos deep learning
        for name, model in self.models.items():
            model.save(f"{base_path}/{name}_model.keras")
            
        # Salvar modelos ML
        for name, model in self.ml_models.items():
            joblib.dump(model, f"{base_path}/{name}_model.pkl")
            
        # Salvar pesos
        joblib.dump(self.weights, f"{base_path}/ensemble_weights.pkl")
        
        print(f"Ensemble salvo em {base_path}")

def prepare_ml_features(customer_ids, product_ids, category_ids, subcategory_ids):
    """Preparar features para modelos ML tradicionais."""
    
    # Criar features simples concatenando os IDs
    features = np.column_stack([
        customer_ids,
        product_ids, 
        category_ids,
        subcategory_ids
    ])
    
    return features

def main():
    """Função principal para criar e treinar o modelo ensemble."""
    
    print("="*70)
    print("CRIANDO MODELO ENSEMBLE DE RECOMENDAÇÃO")
    print("="*70)
    
    # Carregar e preparar os dados
    print("Carregando dados...")
    df = pd.read_parquet(DADOS_TRATADOS)
    
    # Filtrar colunas relevantes
    df = df[['Customer Name', 'Product ID', 'Product Name', 'Sales', 'Category', 'Sub-Category']]
    
    print(f"Dados carregados: {df.shape}")
    
    # Codificar variáveis categóricas
    print("Codificando variáveis categóricas...")
    
    customer_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    subcategory_encoder = LabelEncoder()
    
    df['Customer ID Enc'] = customer_encoder.fit_transform(df['Customer Name'])
    df['Product ID Enc'] = product_encoder.fit_transform(df['Product ID'])
    df['Category Enc'] = category_encoder.fit_transform(df['Category'])
    df['Sub-Category Enc'] = subcategory_encoder.fit_transform(df['Sub-Category'])
    
    # Normalizar vendas
    scaler = MinMaxScaler()
    df['Sales Normalized'] = scaler.fit_transform(df[['Sales']])
    
    # Preparar dados
    customer_ids = df['Customer ID Enc'].values
    product_ids = df['Product ID Enc'].values
    category_ids = df['Category Enc'].values
    subcategory_ids = df['Sub-Category Enc'].values
    sales = df['Sales Normalized'].values
    
    # Dimensões
    num_customers = len(customer_encoder.classes_)
    num_products = len(product_encoder.classes_)
    num_categories = len(category_encoder.classes_)
    num_subcategories = len(subcategory_encoder.classes_)
    
    print(f"Dimensões:")
    print(f"Clientes: {num_customers}")
    print(f"Produtos: {num_products}")
    print(f"Categorias: {num_categories}")
    print(f"Sub-categorias: {num_subcategories}")
    
    # Dividir dados
    (
        customer_train, customer_test,
        product_train, product_test,
        category_train, category_test,
        subcategory_train, subcategory_test,
        sales_train, sales_test
    ) = train_test_split(
        customer_ids, product_ids, category_ids, subcategory_ids, sales,
        test_size=0.2, random_state=42
    )
    
    # Preparar dados para deep learning
    X_train_dl = [
        customer_train.reshape(-1, 1),
        product_train.reshape(-1, 1),
        category_train.reshape(-1, 1),
        subcategory_train.reshape(-1, 1)
    ]
    
    X_test_dl = [
        customer_test.reshape(-1, 1),
        product_test.reshape(-1, 1),
        category_test.reshape(-1, 1),
        subcategory_test.reshape(-1, 1)
    ]
    
    # Preparar dados para ML tradicional
    X_train_ml = prepare_ml_features(customer_train, product_train, category_train, subcategory_train)
    X_test_ml = prepare_ml_features(customer_test, product_test, category_test, subcategory_test)
    
    print(f"Dados de treino: {len(sales_train)}")
    print(f"Dados de teste: {len(sales_test)}")
    
    # Criar modelo ensemble
    print("\n" + "="*50)
    print("CRIANDO MODELO ENSEMBLE")
    print("="*50)
    
    ensemble = EnsembleRecommendationModel()
    
    # Adicionar modelos deep learning
    print("Adicionando modelos deep learning...")
    ensemble.add_deep_model(num_customers, num_products, num_categories, num_subcategories)
    ensemble.add_wide_model(num_customers, num_products, num_categories, num_subcategories)
    ensemble.add_attention_model(num_customers, num_products, num_categories, num_subcategories)
    
    # Adicionar modelos ML
    print("Adicionando modelos de machine learning...")
    ensemble.add_ml_models(X_train_ml)
    
    # Treinar ensemble
    print("\n" + "="*50)
    print("TREINANDO MODELO ENSEMBLE")
    print("="*50)
    
    validation_data = (X_test_dl, X_test_ml, sales_test)
    
    ensemble.train(
        X_train_dl, X_train_ml, sales_train,
        validation_data=validation_data,
        epochs=15
    )
    
    # Avaliar ensemble
    print("\n" + "="*50)
    print("AVALIANDO MODELO ENSEMBLE")
    print("="*50)
    
    # Predições do ensemble
    ensemble_predictions = ensemble.predict(X_test_dl, X_test_ml)
    
    # Métricas
    mse = mean_squared_error(sales_test, ensemble_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(sales_test, ensemble_predictions)
    r2 = r2_score(sales_test, ensemble_predictions)
    
    print("\n" + "="*60)
    print("MÉTRICAS DO MODELO ENSEMBLE:")
    print("="*60)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print("="*60)
    
    # Comparar com modelos individuais
    print("\n" + "="*50)
    print("COMPARAÇÃO COM MODELOS INDIVIDUAIS")
    print("="*50)
    
    individual_results = {}
    
    # Avaliar modelos DL individuais
    for name, model in ensemble.models.items():
        pred = model.predict(X_test_dl, verbose=0).flatten()
        mse_individual = mean_squared_error(sales_test, pred)
        individual_results[name] = mse_individual
        print(f"{name.upper()} MSE: {mse_individual:.6f}")
        
    # Avaliar modelos ML individuais
    for name, model in ensemble.ml_models.items():
        pred = model.predict(X_test_ml)
        mse_individual = mean_squared_error(sales_test, pred)
        individual_results[name] = mse_individual
        print(f"{name.upper()} MSE: {mse_individual:.6f}")
        
    print(f"\nENSEMBLE MSE: {mse:.6f}")
    
    # Verificar se ensemble é melhor
    best_individual = min(individual_results.values())
    improvement = ((best_individual - mse) / best_individual) * 100
    
    print(f"Melhor modelo individual MSE: {best_individual:.6f}")
    print(f"Melhoria do ensemble: {improvement:.2f}%")
    
    # Salvar ensemble
    print("\n" + "="*50)
    print("SALVANDO MODELO ENSEMBLE")
    print("="*50)
    
    ensemble.save("models/ensemble")
    
    # Salvar encoders
    joblib.dump(customer_encoder, "models/ensemble/customer_encoder.pkl")
    joblib.dump(product_encoder, "models/ensemble/product_encoder.pkl")
    joblib.dump(category_encoder, "models/ensemble/category_encoder.pkl")
    joblib.dump(subcategory_encoder, "models/ensemble/subcategory_encoder.pkl")
    joblib.dump(scaler, "models/ensemble/sales_scaler.pkl")
    
    print("Encoders salvos com sucesso!")
    
    # Teste de recomendações com ensemble
    print("\n" + "="*50)
    print("TESTE DE RECOMENDAÇÕES COM ENSEMBLE")
    print("="*50)
    
    def recomendar_produtos_ensemble(customer_name, df, ensemble, encoders, num_products=7):
        """Função para recomendar produtos usando ensemble."""
        
        customer_encoder, product_encoder, category_encoder, subcategory_encoder, scaler = encoders
        
        try:
            customer_id_enc = customer_encoder.transform([customer_name])[0]
        except ValueError:
            return f"Cliente '{customer_name}' não encontrado na base de dados."
        
        # Obter todos os produtos únicos
        unique_products = df[['Product ID', 'Product Name', 'Category', 'Sub-Category']].drop_duplicates()
        
        # Preparar dados para predição
        num_unique_products = len(unique_products)
        customer_array = np.full(num_unique_products, customer_id_enc)
        
        # Codificar produtos, categorias e subcategorias
        product_ids_enc = product_encoder.transform(unique_products['Product ID'])
        category_ids_enc = category_encoder.transform(unique_products['Category'])
        subcategory_ids_enc = subcategory_encoder.transform(unique_products['Sub-Category'])
        
        # Preparar dados para DL
        X_dl = [
            customer_array.reshape(-1, 1),
            product_ids_enc.reshape(-1, 1),
            category_ids_enc.reshape(-1, 1),
            subcategory_ids_enc.reshape(-1, 1)
        ]
        
        # Preparar dados para ML
        X_ml = prepare_ml_features(customer_array, product_ids_enc, category_ids_enc, subcategory_ids_enc)
        
        # Fazer predições com ensemble
        predictions = ensemble.predict(X_dl, X_ml)
        
        # Criar DataFrame com predições
        recommendations_df = unique_products.copy()
        recommendations_df['Prediction'] = predictions
        
        # Filtrar produtos já comprados pelo cliente
        purchased_products = df[df['Customer Name'] == customer_name]['Product ID'].unique()
        recommendations_df = recommendations_df[~recommendations_df['Product ID'].isin(purchased_products)]
        
        # Ordenar por predição e pegar top N
        top_recommendations = recommendations_df.nlargest(num_products, 'Prediction')
        
        # Formatar resultado
        result = top_recommendations[['Product ID', 'Product Name', 'Category', 'Sub-Category']].reset_index(drop=True)
        result.insert(0, 'Ranking', range(1, len(result) + 1))
        
        return result
    
    # Testar recomendações
    cliente_teste = "Darrin Van Huff"
    encoders = (customer_encoder, product_encoder, category_encoder, subcategory_encoder, scaler)
    
    recomendacoes_ensemble = recomendar_produtos_ensemble(
        cliente_teste, df, ensemble, encoders, num_products=7
    )
    
    print(f"\n{'='*70}")
    print(f"RECOMENDAÇÕES ENSEMBLE PARA: {cliente_teste}")
    print(f"{'='*70}")
    print(recomendacoes_ensemble)
    print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("MODELO ENSEMBLE CRIADO COM SUCESSO!")
    print("="*70)
    print("Modelos incluídos no ensemble:")
    print("1. Deep Neural Network (arquitetura profunda)")
    print("2. Wide Neural Network (múltiplas branches)")
    print("3. Attention-based Neural Network")
    print("4. Random Forest Regressor")
    print("5. Gradient Boosting Regressor")
    print("6. Ridge Regression")
    print("="*70)

if __name__ == "__main__":
    main()