"""
Configuração do MLflow para rastreamento de experimentos do modelo de recomendação.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(experiment_name="recomendacao_produtos", tracking_uri=None):
    """
    Configura o MLflow para rastreamento de experimentos.
    
    Args:
        experiment_name: Nome do experimento
        tracking_uri: URI do servidor MLflow (se None, usa local)
    
    Returns:
        ID do experimento
    """
    # Configurar URI de rastreamento
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Usar diretório local se não for especificado
        mlflow_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    # Criar ou obter experimento
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"version": "1.0", "priority": "high"}
        )
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow configurado com sucesso. Experimento: {experiment_name}, ID: {experiment_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    return experiment_id

def log_model_metrics(model, metrics, params, artifacts=None, model_name="recommendation_model"):
    """
    Registra métricas, parâmetros e artefatos do modelo no MLflow.
    
    Args:
        model: Modelo treinado
        metrics: Dicionário com métricas do modelo
        params: Dicionário com parâmetros do modelo
        artifacts: Dicionário com caminhos para artefatos
        model_name: Nome do modelo para registro
    
    Returns:
        run_id: ID da execução MLflow
    """
    with mlflow.start_run() as run:
        # Registrar parâmetros
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Registrar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Registrar artefatos
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(artifact_path, artifact_name)
        
        # Registrar modelo
        mlflow.tensorflow.log_model(model, "model")
        
        # Registrar modelo no registro de modelos
        try:
            mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
        except mlflow.exceptions.MlflowException as e:
            print(f"Aviso ao registrar modelo: {e}")
            # Criar nova versão se o modelo já existir
            client = MlflowClient()
            try:
                client.create_model_version(
                    name=model_name,
                    source=f"runs:/{run.info.run_id}/model",
                    run_id=run.info.run_id
                )
            except Exception as e:
                print(f"Erro ao criar versão do modelo: {e}")
        
        return run.info.run_id

def load_registered_model(model_name="recommendation_model", stage="Production"):
    """
    Carrega um modelo registrado do MLflow.
    
    Args:
        model_name: Nome do modelo registrado
        stage: Estágio do modelo (None, "Staging", "Production", "Archived")
    
    Returns:
        Modelo carregado
    """
    try:
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
        
        model = mlflow.tensorflow.load_model(model_uri)
        print(f"Modelo {model_name} (estágio: {stage}) carregado com sucesso.")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo {model_name}: {e}")
        return None

def compare_models(run_ids, metric_name="mse"):
    """
    Compara modelos com base em uma métrica específica.
    
    Args:
        run_ids: Lista de IDs de execuções MLflow
        metric_name: Nome da métrica para comparação
    
    Returns:
        DataFrame com comparação dos modelos
    """
    import pandas as pd
    
    client = MlflowClient()
    results = []
    
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            if metric_name in metrics:
                result = {
                    "run_id": run_id,
                    metric_name: metrics[metric_name],
                    **params
                }
                results.append(result)
        except Exception as e:
            print(f"Erro ao obter execução {run_id}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        # Ordenar por métrica (assumindo que menor é melhor, como MSE)
        df = df.sort_values(by=metric_name)
        return df
    else:
        print("Nenhum resultado encontrado para comparação.")
        return None

def promote_model_to_production(run_id, model_name="recommendation_model"):
    """
    Promove um modelo para o estágio de produção.
    
    Args:
        run_id: ID da execução MLflow
        model_name: Nome do modelo registrado
    
    Returns:
        Versão do modelo promovido
    """
    client = MlflowClient()
    
    try:
        # Verificar se o modelo já está registrado
        model_versions = client.get_latest_versions(model_name)
        
        # Se não estiver registrado, registrar
        if not model_versions:
            result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
            version = result.version
        else:
            # Verificar se este run_id já está registrado
            run_id_registered = False
            for mv in model_versions:
                if mv.run_id == run_id:
                    version = mv.version
                    run_id_registered = True
                    break
            
            # Se não estiver, registrar nova versão
            if not run_id_registered:
                result = client.create_model_version(
                    name=model_name,
                    source=f"runs:/{run_id}/model",
                    run_id=run_id
                )
                version = result.version
        
        # Promover para produção
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"Modelo {model_name} versão {version} promovido para produção.")
        return version
    
    except Exception as e:
        print(f"Erro ao promover modelo para produção: {e}")
        return None