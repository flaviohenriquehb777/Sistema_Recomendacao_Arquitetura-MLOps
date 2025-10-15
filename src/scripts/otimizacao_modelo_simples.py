#!/usr/bin/env python3
"""
Script simplificado para otimização avançada do modelo de recomendação.
Implementa técnicas avançadas para maximizar a performance do modelo sem MLflow.
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importar utilitários personalizados
from src.config.paths import DADOS_BRUTOS, DADOS_TRATADOS
from src.config.auxiliares_ml import downcast_dataframe

def create_model_with_regularization(num_customers, num_products, num_categories, num_subcategories, 
                                   embedding_dim=50, l2_strength=0.001, dropout_rate=0.2, learning_rate=0.001):
    """Criar modelo com regularização."""
    
    # Inputs
    customer_input = tf.keras.layers.Input(shape=(1,), name='customer_input')
    product_input = tf.keras.layers.Input(shape=(1,), name='product_input')
    category_input = tf.keras.layers.Input(shape=(1,), name='category_input')
    subcategory_input = tf.keras.layers.Input(shape=(1,), name='subcategory_input')
    
    # Embeddings com regularização L2
    customer_embedding = tf.keras.layers.Embedding(
        num_customers, embedding_dim, 
        embeddings_regularizer=tf.keras.regularizers.l2(l2_strength),
        name='customer_embedding'
    )(customer_input)
    
    product_embedding = tf.keras.layers.Embedding(
        num_products, embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_strength),
        name='product_embedding'
    )(product_input)
    
    category_embedding = tf.keras.layers.Embedding(
        num_categories, embedding_dim//2,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_strength),
        name='category_embedding'
    )(category_input)
    
    subcategory_embedding = tf.keras.layers.Embedding(
        num_subcategories, embedding_dim//2,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_strength),
        name='subcategory_embedding'
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
    
    # Camadas densas com dropout
    dense1 = tf.keras.layers.Dense(128, activation='relu', 
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(concat)
    dropout1 = tf.keras.layers.Dropout(dropout_rate)(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(dropout1)
    dropout2 = tf.keras.layers.Dropout(dropout_rate)(dense2)
    
    dense3 = tf.keras.layers.Dense(32, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(dropout2)
    
    # Output
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense3)
    
    # Criar modelo
    model = tf.keras.Model(
        inputs=[customer_input, product_input, category_input, subcategory_input],
        outputs=output
    )
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def hyperparameter_tuning_simple(X, y, num_customers, num_products, num_categories, num_subcategories):
    """Otimização simplificada de hiperparâmetros."""
    
    print("Iniciando otimização de hiperparâmetros...")
    
    # Parâmetros para testar
    param_grid = {
        'embedding_dim': [32, 50, 64],
        'l2_strength': [0.0001, 0.001, 0.01],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.0005, 0.001]
    }
    
    best_score = float('inf')
    best_params = {}
    
    # Split dos dados
    customer_ids, product_ids, category_ids, subcategory_ids = X
    X_train, X_val, y_train, y_val = train_test_split(
        list(zip(customer_ids, product_ids, category_ids, subcategory_ids)), 
        y, test_size=0.2, random_state=42
    )
    
    X_train = list(zip(*X_train))
    X_val = list(zip(*X_val))
    
    # Testar combinações (amostra reduzida para velocidade)
    import itertools
    param_combinations = list(itertools.product(
        param_grid['embedding_dim'][:2],  # Reduzir combinações
        param_grid['l2_strength'][:2],
        param_grid['dropout_rate'][:2],
        param_grid['learning_rate'][:2]
    ))
    
    print(f"Testando {len(param_combinations)} combinações de hiperparâmetros...")
    
    for i, (embedding_dim, l2_strength, dropout_rate, learning_rate) in enumerate(param_combinations):
        print(f"Testando combinação {i+1}/{len(param_combinations)}: "
              f"emb_dim={embedding_dim}, l2={l2_strength}, dropout={dropout_rate}, lr={learning_rate}")
        
        try:
            # Criar modelo
            model = create_model_with_regularization(
                num_customers, num_products, num_categories, num_subcategories,
                embedding_dim=embedding_dim,
                l2_strength=l2_strength,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Treinar rapidamente
            history = model.fit(
                [np.array(X_train[0]).reshape(-1, 1), np.array(X_train[1]).reshape(-1, 1),
                 np.array(X_train[2]).reshape(-1, 1), np.array(X_train[3]).reshape(-1, 1)],
                y_train,
                epochs=10,  # Poucas épocas para velocidade
                batch_size=64,
                validation_data=(
                    [np.array(X_val[0]).reshape(-1, 1), np.array(X_val[1]).reshape(-1, 1),
                     np.array(X_val[2]).reshape(-1, 1), np.array(X_val[3]).reshape(-1, 1)],
                    y_val
                ),
                verbose=0
            )
            
            # Avaliar
            val_loss = min(history.history['val_loss'])
            
            if val_loss < best_score:
                best_score = val_loss
                best_params = {
                    'embedding_dim': embedding_dim,
                    'l2_strength': l2_strength,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate
                }
                print(f"Nova melhor combinação encontrada! Val Loss: {val_loss:.6f}")
            
            # Limpar memória
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"Erro na combinação {i+1}: {e}")
            continue
    
    return best_params

def calculate_recommendation_metrics_simple(model, customer_ids, product_ids, category_ids, subcategory_ids, 
                                          actual_purchases, product_encoder, top_k=7):
    """Calcular métricas de recomendação simplificadas."""
    
    print("Calculando métricas de recomendação...")
    
    # Usar amostra menor para acelerar
    sample_size = min(100, len(customer_ids))
    sample_indices = np.random.choice(len(customer_ids), sample_size, replace=False)
    
    precision_scores = []
    recall_scores = []
    
    for idx in sample_indices:
        customer_id = customer_ids[idx]
        
        if customer_id not in actual_purchases:
            continue
            
        # Simular recomendações para este cliente
        # (implementação simplificada)
        actual_items = set(actual_purchases[customer_id])
        
        if len(actual_items) == 0:
            continue
            
        # Para simplificar, usar produtos aleatórios como recomendações
        all_products = list(product_encoder.classes_)
        recommended_items = set(np.random.choice(all_products, min(top_k, len(all_products)), replace=False))
        
        # Calcular precision e recall
        intersection = len(actual_items.intersection(recommended_items))
        precision = intersection / len(recommended_items) if len(recommended_items) > 0 else 0
        recall = intersection / len(actual_items) if len(actual_items) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        f'precision@{top_k}': avg_precision,
        f'recall@{top_k}': avg_recall,
        f'f1@{top_k}': f1_score
    }

def main():
    """Função principal para executar a otimização do modelo."""
    
    print("="*70)
    print("INICIANDO OTIMIZAÇÃO AVANÇADA DO MODELO DE RECOMENDAÇÃO")
    print("="*70)
    
    # Carregar e preparar os dados
    print("Carregando dados...")
    df = pd.read_parquet(DADOS_TRATADOS)
    
    # Filtrar colunas relevantes
    df = df[['Customer Name', 'Product ID', 'Product Name', 'Sales', 'Category', 'Sub-Category']]
    
    print(f"Dados carregados: {df.shape}")
    print(f"Clientes únicos: {df['Customer Name'].nunique()}")
    print(f"Produtos únicos: {df['Product ID'].nunique()}")
    
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
    
    # Preparar dados para o modelo
    customer_ids = df['Customer ID Enc'].values
    product_ids = df['Product ID Enc'].values
    category_ids = df['Category Enc'].values
    subcategory_ids = df['Sub-Category Enc'].values
    sales = df['Sales Normalized'].values
    
    # Dimensões dos embeddings
    num_customers = len(customer_encoder.classes_)
    num_products = len(product_encoder.classes_)
    num_categories = len(category_encoder.classes_)
    num_subcategories = len(subcategory_encoder.classes_)
    
    print(f"Dimensões dos embeddings:")
    print(f"Clientes: {num_customers}")
    print(f"Produtos: {num_products}")
    print(f"Categorias: {num_categories}")
    print(f"Sub-categorias: {num_subcategories}")
    
    # Preparar dados para validação cruzada
    X = (customer_ids, product_ids, category_ids, subcategory_ids)
    y = sales
    
    # 1. Otimização de Hiperparâmetros
    print("\n" + "="*50)
    print("1. OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("="*50)
    
    best_params = hyperparameter_tuning_simple(
        X, y, num_customers, num_products, num_categories, num_subcategories
    )
    
    print("\n" + "="*50)
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS:")
    print("="*50)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print("="*50)
    
    # 2. Treinamento do Modelo Final Otimizado
    print("\n" + "="*50)
    print("2. TREINAMENTO DO MODELO FINAL OTIMIZADO")
    print("="*50)
    
    final_model = create_model_with_regularization(
        num_customers, num_products, num_categories, num_subcategories,
        **best_params
    )
    
    print("Modelo criado com sucesso!")
    
    # Dividir dados para treinamento final
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
    
    print(f"Dados de treino: {len(customer_train)}")
    print(f"Dados de teste: {len(customer_test)}")
    
    # Treinar modelo final
    print("Treinando modelo final...")
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    # Treinar
    history = final_model.fit(
        [customer_train.reshape(-1, 1), product_train.reshape(-1, 1), 
         category_train.reshape(-1, 1), subcategory_train.reshape(-1, 1)],
        sales_train,
        epochs=30,
        batch_size=32,
        validation_data=(
            [customer_test.reshape(-1, 1), product_test.reshape(-1, 1),
             category_test.reshape(-1, 1), subcategory_test.reshape(-1, 1)],
            sales_test
        ),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("Treinamento concluído!")
    
    # 3. Avaliação do Modelo
    print("\n" + "="*50)
    print("3. AVALIAÇÃO DO MODELO")
    print("="*50)
    
    # Predições no conjunto de teste
    test_predictions = final_model.predict([
        customer_test.reshape(-1, 1),
        product_test.reshape(-1, 1),
        category_test.reshape(-1, 1),
        subcategory_test.reshape(-1, 1)
    ], verbose=0).flatten()
    
    # Métricas
    mse = mean_squared_error(sales_test, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(sales_test, test_predictions)
    r2 = r2_score(sales_test, test_predictions)
    
    print("\n" + "="*60)
    print("MÉTRICAS FINAIS DO MODELO OTIMIZADO:")
    print("="*60)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print("="*60)
    
    # 4. Salvar Modelo Otimizado
    print("\n" + "="*50)
    print("4. SALVANDO MODELO OTIMIZADO")
    print("="*50)
    
    # Criar diretório se não existir
    os.makedirs("models", exist_ok=True)
    
    # Salvar modelo otimizado
    model_path = "models/best_model_recomendacao_otimizado.keras"
    final_model.save(model_path)
    print(f"Modelo salvo em: {model_path}")
    
    # Salvar encoders
    import joblib
    
    joblib.dump(customer_encoder, "models/customer_encoder_otimizado.pkl")
    joblib.dump(product_encoder, "models/product_encoder_otimizado.pkl")
    joblib.dump(category_encoder, "models/category_encoder_otimizado.pkl")
    joblib.dump(subcategory_encoder, "models/subcategory_encoder_otimizado.pkl")
    joblib.dump(scaler, "models/sales_scaler_otimizado.pkl")
    
    print("Encoders e scaler salvos com sucesso!")
    
    # 5. Teste de Recomendações
    print("\n" + "="*50)
    print("5. TESTE DE RECOMENDAÇÕES")
    print("="*50)
    
    def recomendar_produtos_otimizado(customer_name, df, model, customer_encoder, product_encoder, num_products=7):
        """Função otimizada para recomendar produtos."""
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
        
        # Fazer predições
        predictions = model.predict([
            customer_array.reshape(-1, 1),
            product_ids_enc.reshape(-1, 1),
            category_ids_enc.reshape(-1, 1),
            subcategory_ids_enc.reshape(-1, 1)
        ], verbose=0).flatten()
        
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
    recomendacoes_otimizadas = recomendar_produtos_otimizado(
        cliente_teste, df, final_model, customer_encoder, product_encoder, num_products=7
    )
    
    print(f"\n{'='*70}")
    print(f"RECOMENDAÇÕES OTIMIZADAS PARA: {cliente_teste}")
    print(f"{'='*70}")
    print(recomendacoes_otimizadas)
    print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*70)
    print("Técnicas Implementadas:")
    print("1. Otimização de Hiperparâmetros")
    print("2. Regularização L2 e Dropout")
    print("3. Early Stopping e Learning Rate Scheduling")
    print("4. Arquitetura de Embeddings Otimizada")
    print("="*70)

if __name__ == "__main__":
    main()