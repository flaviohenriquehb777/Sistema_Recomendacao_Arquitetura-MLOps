#!/usr/bin/env python3
"""
Script para executar um experimento ao vivo com tracking MLflow
"""

import os
import sys
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Adicionar src ao path
sys.path.append('src')
from config.dagshub_config import setup_dagshub_mlflow

def create_sample_data():
    """
    Cria dados de exemplo para demonstração
    """
    np.random.seed(42)
    
    # Simular dados de recomendação
    n_samples = 1000
    n_customers = 100
    n_products = 50
    
    data = {
        'customer_id': np.random.randint(1, n_customers + 1, n_samples),
        'product_id': np.random.randint(1, n_products + 1, n_samples),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], n_samples),
        'price': np.random.uniform(10, 500, n_samples),
        'rating': np.random.uniform(1, 5, n_samples),
        'sales_quantity': np.random.poisson(3, n_samples) + 1
    }
    
    df = pd.DataFrame(data)
    
    # Criar target (vendas futuras baseadas em rating e preço)
    df['target'] = (df['rating'] * 0.3 + (500 - df['price']) / 100 * 0.2 + 
                   np.random.normal(0, 0.1, n_samples))
    
    return df

def run_live_experiment():
    """
    Executa um experimento ao vivo com tracking completo
    """
    
    # Configurar DagsHub/MLflow
    setup_dagshub_mlflow("experimento_ao_vivo")
    
    print("🚀 Iniciando experimento ao vivo...")
    
    with mlflow.start_run(run_name=f"experimento_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        # 1. Preparar dados
        print("📊 Preparando dados...")
        df = create_sample_data()
        
        # Log dataset info
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("n_features", len(df.columns) - 1)
        mlflow.log_param("n_customers", df['customer_id'].nunique())
        mlflow.log_param("n_products", df['product_id'].nunique())
        
        # 2. Preprocessing
        print("🔧 Preprocessando dados...")
        
        # Encoding categórico
        le_category = LabelEncoder()
        df['category_encoded'] = le_category.fit_transform(df['category'])
        
        # Features e target
        features = ['customer_id', 'product_id', 'category_encoded', 'price', 'rating', 'sales_quantity']
        X = df[features]
        y = df['target']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("scaling", "StandardScaler")
        
        # 3. Criar e treinar modelo
        print("🤖 Criando e treinando modelo...")
        
        model_params = {
            "layers": 3,
            "neurons": [64, 32, 16],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "dropout": 0.3
        }
        
        # Log parâmetros do modelo
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Criar modelo TensorFlow
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callback para logging de métricas durante treinamento
        class MLflowCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    mlflow.log_metric("train_loss", logs.get('loss'), step=epoch)
                    mlflow.log_metric("train_mae", logs.get('mae'), step=epoch)
                    if 'val_loss' in logs:
                        mlflow.log_metric("val_loss", logs.get('val_loss'), step=epoch)
                        mlflow.log_metric("val_mae", logs.get('val_mae'), step=epoch)
        
        # Treinar modelo
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=model_params["epochs"],
            batch_size=model_params["batch_size"],
            callbacks=[MLflowCallback()],
            verbose=1
        )
        
        # 4. Avaliar modelo
        print("📈 Avaliando modelo...")
        
        # Predições
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Métricas finais
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Log métricas finais
        mlflow.log_metric("final_train_mse", train_mse)
        mlflow.log_metric("final_test_mse", test_mse)
        mlflow.log_metric("final_train_mae", train_mae)
        mlflow.log_metric("final_test_mae", test_mae)
        mlflow.log_metric("overfitting_ratio", test_mse / train_mse)
        
        # 5. Salvar artefatos
        print("💾 Salvando artefatos...")
        
        # Salvar modelo
        model.save("temp_model.h5")
        mlflow.log_artifact("temp_model.h5", "model")
        
        # Salvar histórico de treinamento
        history_df = pd.DataFrame(history.history)
        history_df.to_csv("temp_training_history.csv", index=False)
        mlflow.log_artifact("temp_training_history.csv", "training")
        
        # Salvar predições de exemplo
        predictions_df = pd.DataFrame({
            'actual': y_test.iloc[:10].values,
            'predicted': y_pred_test[:10].flatten()
        })
        predictions_df.to_csv("temp_sample_predictions.csv", index=False)
        mlflow.log_artifact("temp_sample_predictions.csv", "predictions")
        
        # 6. Tags e metadados
        mlflow.set_tag("experiment_type", "live_demo")
        mlflow.set_tag("model_framework", "tensorflow")
        mlflow.set_tag("status", "completed")
        mlflow.set_tag("created_by", "automated_script")
        mlflow.set_tag("timestamp", datetime.now().isoformat())
        
        # Limpar arquivos temporários
        for temp_file in ["temp_model.h5", "temp_training_history.csv", "temp_sample_predictions.csv"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"✅ Experimento concluído!")
        print(f"📊 MSE Test: {test_mse:.4f}")
        print(f"📊 MAE Test: {test_mae:.4f}")
        print(f"🔗 Run ID: {run.info.run_id}")

if __name__ == "__main__":
    print("🔄 Executando experimento ao vivo com tracking MLflow...")
    
    try:
        run_live_experiment()
        
        print("\n🎯 Experimento ao vivo concluído com sucesso!")
        print("📊 Verifique os resultados no DagsHub:")
        print("🔗 https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps/experiments")
        
    except Exception as e:
        print(f"❌ Erro durante o experimento: {e}")
        import traceback
        traceback.print_exc()
