"""
Módulo para configuração e utilização do MLflow para rastreamento de experimentos.
"""
import os
import mlflow
from mlflow.models import infer_signature
import tensorflow as tf

# Configuração do servidor MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Pode ser alterado para o servidor DagsHub

def setup_mlflow():
    """Configura o MLflow com o URI de rastreamento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow configurado com URI: {MLFLOW_TRACKING_URI}")
    return mlflow.get_tracking_uri()

def log_model_metrics(model, X_test, y_test, metrics_dict, model_name="recommendation_model"):
    """
    Registra métricas e o modelo no MLflow.
    
    Args:
        model: Modelo treinado (TensorFlow/Keras)
        X_test: Dados de teste (features)
        y_test: Dados de teste (target)
        metrics_dict: Dicionário com métricas de avaliação
        model_name: Nome do modelo para registro
    """
    with mlflow.start_run(run_name=model_name) as run:
        # Registrar parâmetros do modelo
        for layer in model.layers:
            if hasattr(layer, 'get_config'):
                config = layer.get_config()
                for key, value in config.items():
                    if isinstance(value, (int, float, str, bool)):
                        mlflow.log_param(f"{layer.name}_{key}", value)
        
        # Registrar métricas
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Registrar o modelo
        signature = infer_signature(X_test, model.predict(X_test))
        
        # Salvar o modelo no formato TensorFlow SavedModel
        mlflow.tensorflow.log_model(
            model, 
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name
        )
        
        print(f"Modelo registrado com sucesso. Run ID: {run.info.run_id}")
        return run.info.run_id

def load_registered_model(model_name="recommendation_model", stage="Production"):
    """
    Carrega um modelo registrado do MLflow.
    
    Args:
        model_name: Nome do modelo registrado
        stage: Estágio do modelo (None, Staging, Production)
        
    Returns:
        Modelo carregado
    """
    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.tensorflow.load_model(model_uri)
        print(f"Modelo {model_name} (estágio: {stage}) carregado com sucesso.")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

if __name__ == "__main__":
    # Teste da configuração
    tracking_uri = setup_mlflow()
    print(f"MLflow Tracking URI: {tracking_uri}")