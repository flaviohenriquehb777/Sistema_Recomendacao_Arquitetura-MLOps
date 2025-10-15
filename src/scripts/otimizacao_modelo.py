#!/usr/bin/env python3
"""
Script para otimização avançada do modelo de recomendação.
Implementa técnicas avançadas para maximizar a performance do modelo.
"""

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import mlflow
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importar utilitários personalizados
from src.config.paths import DADOS_BRUTOS, DADOS_TRATADOS
from src.config.auxiliares_ml import downcast_dataframe
from src.config.model_utils import (
    create_model_with_regularization,
    cross_validate_model,
    calculate_recommendation_metrics,
    hyperparameter_tuning
)
from mlflow_tracking import setup_mlflow, log_model_metrics

def main():
    """Função principal para executar a otimização do modelo."""
    
    print("="*70)
    print("INICIANDO OTIMIZAÇÃO AVANÇADA DO MODELO DE RECOMENDAÇÃO")
    print("="*70)
    
    # Configurar MLflow
    print("Configurando MLflow...")
    setup_mlflow()
    
    # Carregar e preparar os dados
    print("Carregando dados...")
    df = pd.read_parquet(DADOS_TRATADOS)
    
    # Filtrar colunas relevantes
    df = df[['Customer Name', 'Product ID', 'Product Name', 'Sales', 'Category', 'Sub-Category']]
    
    print(f"Dados carregados: {df.shape}")
    print(f"Clientes únicos: {df['Customer Name'].nunique()}")
    print(f"Produtos únicos: {df['Product ID'].nunique()}")
    print(f"Categorias únicas: {df['Category'].nunique()}")
    print(f"Sub-categorias únicas: {df['Sub-Category'].nunique()}")
    
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
    print("Iniciando otimização de hiperparâmetros...")
    print("Isso pode levar alguns minutos...")
    
    best_params = hyperparameter_tuning(
        X, y, num_customers, num_products, num_categories, num_subcategories
    )
    
    print("\n" + "="*50)
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS:")
    print("="*50)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print("="*50)
    
    # 2. Validação Cruzada com Melhores Parâmetros
    print("\n" + "="*50)
    print("2. VALIDAÇÃO CRUZADA COM MELHORES PARÂMETROS")
    print("="*50)
    print("Executando validação cruzada com os melhores parâmetros...")
    
    cv_results = cross_validate_model(
        X, y, 
        num_customers, num_products, num_categories, num_subcategories,
        embedding_dim=best_params.get('embedding_dim', 50),
        n_splits=5,
        epochs=25,
        batch_size=32
    )
    
    print("\n" + "="*50)
    print("RESULTADOS DA VALIDAÇÃO CRUZADA:")
    print("="*50)
    print(f"MSE Médio: {cv_results['mean_mse']:.6f} ± {cv_results['std_mse']:.6f}")
    print(f"Val Loss Médio: {cv_results['mean_val_loss']:.6f} ± {cv_results['std_val_loss']:.6f}")
    print("\nResultados por fold:")
    for fold_result in cv_results['fold_metrics']:
        print(f"Fold {fold_result['fold']}: MSE = {fold_result['mse']:.6f}, Val Loss = {fold_result['val_loss']:.6f}")
    print("="*50)
    
    # 3. Treinamento do Modelo Final Otimizado
    print("\n" + "="*50)
    print("3. TREINAMENTO DO MODELO FINAL OTIMIZADO")
    print("="*50)
    print("Criando modelo final otimizado...")
    
    final_model = create_model_with_regularization(
        num_customers, num_products, num_categories, num_subcategories,
        embedding_dim=best_params.get('embedding_dim', 50),
        l2_strength=best_params.get('l2_strength', 0.001),
        dropout_rate=best_params.get('dropout_rate', 0.2)
    )
    
    # Ajustar learning rate
    final_model.optimizer.learning_rate = best_params.get('learning_rate', 0.0005)
    
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
    
    # TensorBoard
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_otimizado"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Treinar
    history = final_model.fit(
        [customer_train.reshape(-1, 1), product_train.reshape(-1, 1), 
         category_train.reshape(-1, 1), subcategory_train.reshape(-1, 1)],
        sales_train,
        epochs=50,
        batch_size=32,
        validation_data=(
            [customer_test.reshape(-1, 1), product_test.reshape(-1, 1),
             category_test.reshape(-1, 1), subcategory_test.reshape(-1, 1)],
            sales_test
        ),
        callbacks=[early_stopping, reduce_lr, tensorboard_callback],
        verbose=1
    )
    
    print("Treinamento concluído!")
    
    # 4. Avaliação Avançada do Modelo
    print("\n" + "="*50)
    print("4. AVALIAÇÃO AVANÇADA DO MODELO")
    print("="*50)
    print("Preparando dados para métricas de recomendação...")
    
    # Criar dicionário de compras reais por cliente
    actual_purchases = {}
    for idx, row in df.iterrows():
        customer_id = row['Customer ID Enc']
        product_id = row['Product ID']
        
        if customer_id not in actual_purchases:
            actual_purchases[customer_id] = []
        
        if product_id not in actual_purchases[customer_id]:
            actual_purchases[customer_id].append(product_id)
    
    print(f"Dados preparados para {len(actual_purchases)} clientes")
    
    # Calcular métricas de recomendação
    print("Calculando métricas de recomendação...")
    
    # Usar uma amostra para acelerar o cálculo
    sample_size = min(1000, len(customer_test))
    sample_indices = np.random.choice(len(customer_test), sample_size, replace=False)
    
    customer_sample = customer_test[sample_indices]
    product_sample = product_test[sample_indices]
    category_sample = category_test[sample_indices]
    subcategory_sample = subcategory_test[sample_indices]
    
    recommendation_metrics = calculate_recommendation_metrics(
        final_model,
        customer_sample,
        product_sample,
        category_sample,
        subcategory_sample,
        actual_purchases,
        product_encoder,
        top_k=7
    )
    
    print("\n" + "="*50)
    print("MÉTRICAS DE RECOMENDAÇÃO:")
    print("="*50)
    for metric_name, metric_value in recommendation_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("="*50)
    
    # Calcular métricas finais do modelo
    print("Calculando métricas finais...")
    
    # Predições no conjunto de teste
    test_predictions = final_model.predict([
        customer_test.reshape(-1, 1),
        product_test.reshape(-1, 1),
        category_test.reshape(-1, 1),
        subcategory_test.reshape(-1, 1)
    ], verbose=0).flatten()
    
    # MSE e RMSE
    mse = mean_squared_error(sales_test, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(sales_test, test_predictions)
    r2 = r2_score(sales_test, test_predictions)
    
    final_metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        **recommendation_metrics
    }
    
    print("\n" + "="*60)
    print("MÉTRICAS FINAIS DO MODELO OTIMIZADO:")
    print("="*60)
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print("\nMétricas de Recomendação:")
    for metric_name, metric_value in recommendation_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("="*60)
    
    # 5. Registro no MLflow
    print("\n" + "="*50)
    print("5. REGISTRO NO MLFLOW")
    print("="*50)
    print("Registrando modelo otimizado no MLflow...")
    
    X_test_formatted = [
        customer_test.reshape(-1, 1),
        product_test.reshape(-1, 1),
        category_test.reshape(-1, 1),
        subcategory_test.reshape(-1, 1)
    ]
    
    run_id = log_model_metrics(
        final_model,
        X_test_formatted,
        sales_test,
        final_metrics,
        model_name="recommendation_model_optimized"
    )
    
    print(f"Modelo registrado com sucesso! Run ID: {run_id}")
    
    # 6. Salvar Modelo Otimizado
    print("\n" + "="*50)
    print("6. SALVANDO MODELO OTIMIZADO")
    print("="*50)
    
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
    
    # 7. Teste de Recomendações
    print("\n" + "="*50)
    print("7. TESTE DE RECOMENDAÇÕES")
    print("="*50)
    
    def recomendar_produtos_otimizado(customer_name, df, model, customer_encoder, product_encoder, num_products=7):
        """
        Função otimizada para recomendar produtos usando o modelo melhorado.
        """
        try:
            # Codificar o nome do cliente
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
    
    # Testar recomendações com o modelo otimizado
    print("Testando recomendações com o modelo otimizado...")
    
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
    print("2. Validação Cruzada")
    print("3. Regularização L2 e Dropout")
    print("4. Early Stopping e Learning Rate Scheduling")
    print("5. Métricas Avançadas de Recomendação")
    print("6. MLflow Tracking")
    print("="*70)

if __name__ == "__main__":
    main()