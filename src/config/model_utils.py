"""
Utilitários avançados para o modelo de recomendação.
Inclui validação cruzada, regularização e outras técnicas para melhorar o desempenho.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.tensorflow

def create_model_with_regularization(num_customers, num_products, num_categories, num_subcategories, 
                                    embedding_dim=50, l2_strength=0.001, dropout_rate=0.2):
    """
    Cria o modelo de recomendação com regularização L2 e dropout para evitar overfitting.
    
    Args:
        num_customers: Número de clientes únicos
        num_products: Número de produtos únicos
        num_categories: Número de categorias únicas
        num_subcategories: Número de subcategorias únicas
        embedding_dim: Dimensão dos embeddings
        l2_strength: Força da regularização L2
        dropout_rate: Taxa de dropout
        
    Returns:
        Modelo compilado
    """
    # Entradas
    customer_input = layers.Input(shape=(1,), name='customer_input')
    product_input = layers.Input(shape=(1,), name='product_input')
    category_input = layers.Input(shape=(1,), name='category_input')
    subcategory_input = layers.Input(shape=(1,), name='subcategory_input')

    # Embeddings com regularização L2
    customer_embeddings = layers.Embedding(
        input_dim=num_customers, 
        output_dim=embedding_dim, 
        embeddings_regularizer=regularizers.l2(l2_strength),
        name='customer_embeddings'
    )(customer_input)
    
    product_embeddings = layers.Embedding(
        input_dim=num_products, 
        output_dim=embedding_dim, 
        embeddings_regularizer=regularizers.l2(l2_strength),
        name='product_embeddings'
    )(product_input)
    
    category_embeddings = layers.Embedding(
        input_dim=num_categories, 
        output_dim=embedding_dim, 
        embeddings_regularizer=regularizers.l2(l2_strength),
        name='category_embeddings'
    )(category_input)
    
    subcategory_embeddings = layers.Embedding(
        input_dim=num_subcategories, 
        output_dim=embedding_dim, 
        embeddings_regularizer=regularizers.l2(l2_strength),
        name='subcategory_embeddings'
    )(subcategory_input)

    # Flatten
    customer_vec = layers.Flatten(name='customer_flatten')(customer_embeddings)
    product_vec = layers.Flatten(name='product_flatten')(product_embeddings)
    category_vec = layers.Flatten(name='category_flatten')(category_embeddings)
    subcategory_vec = layers.Flatten(name='subcategory_flatten')(subcategory_embeddings)

    # Concatenação
    concat_vec = layers.Concatenate(name='concat')([customer_vec, product_vec, category_vec, subcategory_vec])
    
    # Camadas densas com regularização e dropout
    dense_1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), name='dense_1')(concat_vec)
    dropout_1 = layers.Dropout(dropout_rate)(dense_1)
    
    dense_2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), name='dense_2')(dropout_1)
    dropout_2 = layers.Dropout(dropout_rate)(dense_2)
    
    dense_3 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), name='dense_3')(dropout_2)
    
    # Saída
    output = layers.Dense(1, activation='linear', name='output')(dense_3)

    # Modelo
    model = tf.keras.Model([customer_input, product_input, category_input, subcategory_input], output)
    
    # Compilação com otimizador Adam e learning rate reduzido
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def cross_validate_model(X, y, num_customers, num_products, num_categories, num_subcategories, 
                        embedding_dim=50, n_splits=5, epochs=20, batch_size=32):
    """
    Realiza validação cruzada do modelo de recomendação.
    
    Args:
        X: Tupla com (customer_ids, product_ids, category_ids, subcategory_ids)
        y: Target (sales)
        num_customers, num_products, num_categories, num_subcategories: Dimensões dos embeddings
        embedding_dim: Dimensão dos embeddings
        n_splits: Número de folds para validação cruzada
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        
    Returns:
        Dicionário com métricas de validação cruzada e melhor modelo
    """
    customer_ids, product_ids, category_ids, subcategory_ids = X
    
    # Preparar validação cruzada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    best_model = None
    best_mse = float('inf')
    
    # Iniciar rastreamento MLflow
    with mlflow.start_run(run_name="cross_validation"):
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        
        # Para cada fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(customer_ids)):
            print(f"Treinando fold {fold+1}/{n_splits}")
            
            # Separar dados de treino e validação
            customer_train, customer_val = customer_ids[train_idx], customer_ids[val_idx]
            product_train, product_val = product_ids[train_idx], product_ids[val_idx]
            category_train, category_val = category_ids[train_idx], category_ids[val_idx]
            subcategory_train, subcategory_val = subcategory_ids[train_idx], subcategory_ids[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Criar e treinar modelo
            model = create_model_with_regularization(
                num_customers, num_products, num_categories, num_subcategories, embedding_dim
            )
            
            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Treinar modelo
            history = model.fit(
                [customer_train, product_train, category_train, subcategory_train],
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([customer_val, product_val, category_val, subcategory_val], y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Avaliar modelo
            val_loss = history.history['val_loss'][-1]
            y_pred = model.predict([customer_val, product_val, category_val, subcategory_val])
            mse = mean_squared_error(y_val, y_pred)
            
            # Registrar métricas
            fold_metrics.append({
                'fold': fold + 1,
                'val_loss': val_loss,
                'mse': mse
            })
            
            mlflow.log_metric(f"fold_{fold+1}_val_loss", val_loss)
            mlflow.log_metric(f"fold_{fold+1}_mse", mse)
            
            # Verificar se é o melhor modelo
            if mse < best_mse:
                best_mse = mse
                best_model = model
        
        # Calcular métricas médias
        mean_val_loss = np.mean([m['val_loss'] for m in fold_metrics])
        mean_mse = np.mean([m['mse'] for m in fold_metrics])
        std_val_loss = np.std([m['val_loss'] for m in fold_metrics])
        std_mse = np.std([m['mse'] for m in fold_metrics])
        
        # Registrar métricas médias
        mlflow.log_metric("mean_val_loss", mean_val_loss)
        mlflow.log_metric("mean_mse", mean_mse)
        mlflow.log_metric("std_val_loss", std_val_loss)
        mlflow.log_metric("std_mse", std_mse)
        
        # Registrar melhor modelo
        mlflow.tensorflow.log_model(best_model, "best_model")
    
    return {
        'fold_metrics': fold_metrics,
        'mean_val_loss': mean_val_loss,
        'mean_mse': mean_mse,
        'std_val_loss': std_val_loss,
        'std_mse': std_mse,
        'best_model': best_model
    }

def calculate_recommendation_metrics(model, customer_ids, product_ids, category_ids, subcategory_ids, 
                                    actual_purchases, product_encoder, top_k=7):
    """
    Calcula métricas de recomendação como precision@k, recall@k e F1@k.
    
    Args:
        model: Modelo treinado
        customer_ids: IDs dos clientes para avaliação
        product_ids, category_ids, subcategory_ids: Dados de produtos
        actual_purchases: Dicionário {customer_id: [lista de product_ids comprados]}
        product_encoder: Encoder para converter IDs de produtos
        top_k: Número de recomendações a serem feitas
        
    Returns:
        Dicionário com métricas de recomendação
    """
    unique_customers = np.unique(customer_ids)
    precision_at_k = []
    recall_at_k = []
    f1_at_k = []
    
    for customer_id in unique_customers:
        # Obter produtos que o cliente realmente comprou
        if customer_id not in actual_purchases:
            continue
            
        actual_products = set(actual_purchases[customer_id])
        if not actual_products:
            continue
            
        # Prever scores para todos os produtos para este cliente
        num_products = len(product_encoder.classes_)
        customer_array = np.full(num_products, customer_id)
        product_array = np.arange(num_products)
        category_array = np.zeros(num_products)  # Simplificado para avaliação
        subcategory_array = np.zeros(num_products)  # Simplificado para avaliação
        
        # Prever scores
        scores = model.predict([
            customer_array.reshape(-1, 1),
            product_array.reshape(-1, 1),
            category_array.reshape(-1, 1),
            subcategory_array.reshape(-1, 1)
        ], verbose=0).flatten()
        
        # Obter top-k produtos recomendados
        top_indices = np.argsort(scores)[::-1][:top_k]
        recommended_products = set(product_encoder.inverse_transform(top_indices))
        
        # Calcular métricas
        true_positives = len(actual_products.intersection(recommended_products))
        precision = true_positives / len(recommended_products) if recommended_products else 0
        recall = true_positives / len(actual_products) if actual_products else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_at_k.append(precision)
        recall_at_k.append(recall)
        f1_at_k.append(f1)
    
    # Calcular médias
    avg_precision = np.mean(precision_at_k) if precision_at_k else 0
    avg_recall = np.mean(recall_at_k) if recall_at_k else 0
    avg_f1 = np.mean(f1_at_k) if f1_at_k else 0
    
    return {
        f'precision@{top_k}': avg_precision,
        f'recall@{top_k}': avg_recall,
        f'f1@{top_k}': avg_f1
    }

def hyperparameter_tuning(X, y, num_customers, num_products, num_categories, num_subcategories):
    """
    Realiza otimização de hiperparâmetros para o modelo de recomendação.
    
    Args:
        X: Tupla com (customer_ids, product_ids, category_ids, subcategory_ids)
        y: Target (sales)
        num_customers, num_products, num_categories, num_subcategories: Dimensões dos embeddings
        
    Returns:
        Dicionário com melhores hiperparâmetros
    """
    # Definir grade de hiperparâmetros
    embedding_dims = [30, 50, 70]
    l2_strengths = [0.0001, 0.001, 0.01]
    dropout_rates = [0.1, 0.2, 0.3]
    learning_rates = [0.001, 0.0005, 0.0001]
    
    # Dividir dados em treino e validação
    from sklearn.model_selection import train_test_split
    
    customer_ids, product_ids, category_ids, subcategory_ids = X
    
    (customer_train, customer_val,
     product_train, product_val,
     category_train, category_val,
     subcategory_train, subcategory_val,
     y_train, y_val) = train_test_split(
        customer_ids, product_ids, category_ids, subcategory_ids, y,
        test_size=0.2, random_state=42
    )
    
    best_val_loss = float('inf')
    best_params = {}
    
    # Iniciar rastreamento MLflow
    with mlflow.start_run(run_name="hyperparameter_tuning"):
        # Testar combinações de hiperparâmetros
        for embedding_dim in embedding_dims:
            for l2_strength in l2_strengths:
                for dropout_rate in dropout_rates:
                    for learning_rate in learning_rates:
                        # Registrar parâmetros
                        run_params = {
                            "embedding_dim": embedding_dim,
                            "l2_strength": l2_strength,
                            "dropout_rate": dropout_rate,
                            "learning_rate": learning_rate
                        }
                        
                        with mlflow.start_run(nested=True, run_name=f"params_{embedding_dim}_{l2_strength}_{dropout_rate}_{learning_rate}"):
                            for param_name, param_value in run_params.items():
                                mlflow.log_param(param_name, param_value)
                            
                            # Criar modelo
                            model = create_model_with_regularization(
                                num_customers, num_products, num_categories, num_subcategories,
                                embedding_dim=embedding_dim,
                                l2_strength=l2_strength,
                                dropout_rate=dropout_rate
                            )
                            
                            # Ajustar learning rate
                            model.optimizer.learning_rate = learning_rate
                            
                            # Callbacks
                            early_stopping = tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=3,
                                restore_best_weights=True
                            )
                            
                            # Treinar modelo
                            history = model.fit(
                                [customer_train, product_train, category_train, subcategory_train],
                                y_train,
                                epochs=15,  # Reduzido para otimização
                                batch_size=32,
                                validation_data=([customer_val, product_val, category_val, subcategory_val], y_val),
                                callbacks=[early_stopping],
                                verbose=0
                            )
                            
                            # Avaliar modelo
                            val_loss = history.history['val_loss'][-1]
                            mlflow.log_metric("val_loss", val_loss)
                            
                            # Verificar se é o melhor modelo
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_params = run_params.copy()
                                
                                # Registrar melhor modelo até agora
                                mlflow.tensorflow.log_model(model, "model")
        
        # Registrar melhores parâmetros
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)
        mlflow.log_metric("best_val_loss", best_val_loss)
    
    return best_params