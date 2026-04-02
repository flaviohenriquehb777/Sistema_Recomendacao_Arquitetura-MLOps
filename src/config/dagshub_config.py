#!/usr/bin/env python3
"""
Configuração para integração com DagsHub e MLflow
"""

import os
import mlflow
import dagshub
from typing import Optional

# Configurações do DagsHub
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "flaviohenriquehb777")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Sistema_Recomendacao_Arquitetura-MLOps")
DAGSHUB_URL = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"

def setup_dagshub_mlflow(experiment_name: str = "sistema_recomendacao") -> bool:
    """
    Configura a integração com DagsHub e MLflow
    
    Args:
        experiment_name (str): Nome do experimento MLflow
    """
    try:
        # Inicializar DagsHub
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True
        )
        
        # Configurar MLflow tracking URI
        mlflow_tracking_uri = f"{DAGSHUB_URL}.mlflow"
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Configurar experimento
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ DagsHub e MLflow configurados com sucesso!")
        print(f"📊 Tracking URI: {mlflow_tracking_uri}")
        print(f"🧪 Experimento: {experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao configurar DagsHub/MLflow: {e}")
        print("⚠️ Continuando sem tracking remoto...")
        return False

def log_model_experiment(
    model_name: str,
    model,
    metrics: dict,
    params: dict,
    artifacts: Optional[dict] = None
) -> None:
    """
    Registra um experimento completo no MLflow
    
    Args:
        model_name (str): Nome do modelo
        model: Modelo treinado
        metrics (dict): Métricas do modelo
        params (dict): Parâmetros do modelo
        artifacts (dict, optional): Artefatos adicionais
    """
    try:
        with mlflow.start_run(run_name=model_name):
            # Log parâmetros
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log métricas
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log modelo
            if hasattr(model, 'save'):  # TensorFlow/Keras
                mlflow.tensorflow.log_model(
                    model, 
                    artifact_path="model",
                    registered_model_name=model_name
                )
            else:  # Scikit-learn
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=model_name
                )
            
            # Log artefatos adicionais
            if artifacts:
                for name, path in artifacts.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, artifact_path=name)
            
            print(f"✅ Experimento '{model_name}' registrado com sucesso!")
            
    except Exception as e:
        print(f"❌ Erro ao registrar experimento: {e}")

def get_best_model_from_experiments(experiment_name: str, metric_name: str = "mse") -> Optional[str]:
    """
    Obtém o melhor modelo de um experimento baseado em uma métrica
    
    Args:
        experiment_name (str): Nome do experimento
        metric_name (str): Nome da métrica para comparação
        
    Returns:
        str: ID do melhor run ou None se não encontrado
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experimento '{experiment_name}' não encontrado")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} ASC"],
            max_results=1
        )
        
        if len(runs) > 0:
            best_run_id = runs.iloc[0]['run_id']
            best_metric = runs.iloc[0][f'metrics.{metric_name}']
            print(f"🏆 Melhor modelo encontrado:")
            print(f"   Run ID: {best_run_id}")
            print(f"   {metric_name.upper()}: {best_metric}")
            return best_run_id
        else:
            print(f"❌ Nenhum run encontrado no experimento '{experiment_name}'")
            return None
            
    except Exception as e:
        print(f"❌ Erro ao buscar melhor modelo: {e}")
        return None

def compare_models_performance(experiment_name: str) -> None:
    """
    Compara a performance de todos os modelos em um experimento
    
    Args:
        experiment_name (str): Nome do experimento
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"❌ Experimento '{experiment_name}' não encontrado")
            return
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mse ASC"]
        )
        
        if len(runs) > 0:
            print(f"📊 COMPARAÇÃO DE MODELOS - {experiment_name}")
            print("=" * 80)
            
            for idx, run in runs.iterrows():
                run_name = run.get('tags.mlflow.runName', 'Unnamed')
                mse = run.get('metrics.mse', 'N/A')
                rmse = run.get('metrics.rmse', 'N/A')
                mae = run.get('metrics.mae', 'N/A')
                
                print(f"🔹 {run_name}")
                print(f"   MSE: {mse}")
                print(f"   RMSE: {rmse}")
                print(f"   MAE: {mae}")
                print("-" * 40)
        else:
            print(f"❌ Nenhum run encontrado no experimento '{experiment_name}'")
            
    except Exception as e:
        print(f"❌ Erro ao comparar modelos: {e}")

if __name__ == "__main__":
    # Teste da configuração
    setup_dagshub_mlflow("teste_configuracao")
    print("🧪 Configuração testada!")
