#!/usr/bin/env python3
"""
Módulo para tracking de experimentos no DagsHub/MLflow
"""

import os
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Classe para gerenciar experimentos no DagsHub/MLflow"""
    
    def __init__(self, 
                 dagshub_repo: str = "flaviohenriquehb777/Projeto_7_Sistema_de_Recomendacao",
                 experiment_name: str = "recommendation_system"):
        """
        Inicializar o tracker de experimentos
        
        Args:
            dagshub_repo: Repositório do DagsHub
            experiment_name: Nome do experimento
        """
        self.dagshub_repo = dagshub_repo
        self.experiment_name = experiment_name
        self.setup_tracking()
        
    def setup_tracking(self):
        """Configurar tracking do DagsHub/MLflow"""
        try:
            # Inicializar DagsHub
            dagshub.init(repo_owner="flaviohenriquehb777", 
                        repo_name="Projeto_7_Sistema_de_Recomendacao", 
                        mlflow=True)
            
            # Configurar MLflow
            mlflow.set_experiment(self.experiment_name)
            
            logger.info(f"✓ Tracking configurado para experimento: {self.experiment_name}")
            
        except Exception as e:
            logger.warning(f"Erro ao configurar tracking: {e}")
            logger.info("Continuando sem tracking remoto...")
    
    def start_run(self, run_name: str = None) -> str:
        """
        Iniciar uma nova execução
        
        Args:
            run_name: Nome da execução
            
        Returns:
            ID da execução
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        mlflow.start_run(run_name=run_name)
        run_id = mlflow.active_run().info.run_id
        
        logger.info(f"✓ Execução iniciada: {run_name} (ID: {run_id})")
        return run_id
    
    def log_params(self, params: Dict[str, Any]):
        """
        Registrar parâmetros do modelo
        
        Args:
            params: Dicionário com parâmetros
        """
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info(f"✓ Parâmetros registrados: {list(params.keys())}")
        except Exception as e:
            logger.error(f"Erro ao registrar parâmetros: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Registrar métricas do modelo
        
        Args:
            metrics: Dicionário com métricas
            step: Passo da métrica (opcional)
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.info(f"✓ Métricas registradas: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Erro ao registrar métricas: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "tensorflow"):
        """
        Registrar modelo
        
        Args:
            model: Modelo treinado
            model_name: Nome do modelo
            model_type: Tipo do modelo (tensorflow, sklearn, etc.)
        """
        try:
            if model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, model_name)
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            else:
                mlflow.log_artifact(model, model_name)
                
            logger.info(f"✓ Modelo registrado: {model_name}")
        except Exception as e:
            logger.error(f"Erro ao registrar modelo: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """
        Registrar artefato
        
        Args:
            artifact_path: Caminho do artefato
            artifact_name: Nome do artefato (opcional)
        """
        try:
            if artifact_name:
                mlflow.log_artifact(artifact_path, artifact_name)
            else:
                mlflow.log_artifact(artifact_path)
            logger.info(f"✓ Artefato registrado: {artifact_path}")
        except Exception as e:
            logger.error(f"Erro ao registrar artefato: {e}")
    
    def log_dataset_info(self, df: pd.DataFrame, dataset_name: str):
        """
        Registrar informações do dataset
        
        Args:
            df: DataFrame com os dados
            dataset_name: Nome do dataset
        """
        try:
            dataset_info = {
                f"{dataset_name}_rows": len(df),
                f"{dataset_name}_columns": len(df.columns),
                f"{dataset_name}_memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                f"{dataset_name}_null_values": df.isnull().sum().sum()
            }
            
            # Adicionar estatísticas das colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                dataset_info[f"{dataset_name}_{col}_mean"] = df[col].mean()
                dataset_info[f"{dataset_name}_{col}_std"] = df[col].std()
                dataset_info[f"{dataset_name}_{col}_min"] = df[col].min()
                dataset_info[f"{dataset_name}_{col}_max"] = df[col].max()
            
            self.log_params(dataset_info)
            logger.info(f"✓ Informações do dataset registradas: {dataset_name}")
            
        except Exception as e:
            logger.error(f"Erro ao registrar informações do dataset: {e}")
    
    def log_training_history(self, history, model_name: str):
        """
        Registrar histórico de treinamento
        
        Args:
            history: Histórico do treinamento (Keras History)
            model_name: Nome do modelo
        """
        try:
            if hasattr(history, 'history'):
                history_dict = history.history
                
                for epoch, (loss, val_loss) in enumerate(zip(
                    history_dict.get('loss', []),
                    history_dict.get('val_loss', [])
                )):
                    self.log_metrics({
                        f"{model_name}_loss": loss,
                        f"{model_name}_val_loss": val_loss
                    }, step=epoch)
                
                # Registrar métricas adicionais se existirem
                for metric_name, values in history_dict.items():
                    if metric_name not in ['loss', 'val_loss']:
                        for epoch, value in enumerate(values):
                            self.log_metrics({
                                f"{model_name}_{metric_name}": value
                            }, step=epoch)
                
                logger.info(f"✓ Histórico de treinamento registrado: {model_name}")
                
        except Exception as e:
            logger.error(f"Erro ao registrar histórico: {e}")
    
    def end_run(self):
        """Finalizar execução atual"""
        try:
            mlflow.end_run()
            logger.info("✓ Execução finalizada")
        except Exception as e:
            logger.error(f"Erro ao finalizar execução: {e}")
    
    def get_experiment_runs(self) -> pd.DataFrame:
        """
        Obter todas as execuções do experimento
        
        Returns:
            DataFrame com informações das execuções
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                logger.info(f"✓ Encontradas {len(runs)} execuções")
                return runs
            else:
                logger.warning("Experimento não encontrado")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter execuções: {e}")
            return pd.DataFrame()
    
    def get_best_run(self, metric_name: str, ascending: bool = True) -> Optional[Dict]:
        """
        Obter a melhor execução baseada em uma métrica
        
        Args:
            metric_name: Nome da métrica
            ascending: Se True, menor valor é melhor
            
        Returns:
            Informações da melhor execução
        """
        try:
            runs_df = self.get_experiment_runs()
            if not runs_df.empty and f"metrics.{metric_name}" in runs_df.columns:
                best_run = runs_df.loc[runs_df[f"metrics.{metric_name}"].idxmin() if ascending 
                                     else runs_df[f"metrics.{metric_name}"].idxmax()]
                
                logger.info(f"✓ Melhor execução encontrada: {best_run['run_id']}")
                return best_run.to_dict()
            else:
                logger.warning(f"Métrica {metric_name} não encontrada")
                return None
        except Exception as e:
            logger.error(f"Erro ao obter melhor execução: {e}")
            return None
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Comparar múltiplas execuções
        
        Args:
            run_ids: Lista de IDs das execuções
            
        Returns:
            DataFrame com comparação das execuções
        """
        try:
            runs_data = []
            for run_id in run_ids:
                run = mlflow.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status
                }
                
                # Adicionar parâmetros
                run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
                
                # Adicionar métricas
                run_data.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
                
                runs_data.append(run_data)
            
            comparison_df = pd.DataFrame(runs_data)
            logger.info(f"✓ Comparação de {len(run_ids)} execuções criada")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Erro ao comparar execuções: {e}")
            return pd.DataFrame()
    
    def export_experiment_report(self, output_path: str = "experiment_report.json"):
        """
        Exportar relatório do experimento
        
        Args:
            output_path: Caminho do arquivo de saída
        """
        try:
            runs_df = self.get_experiment_runs()
            
            if not runs_df.empty:
                # Criar relatório
                report = {
                    "experiment_name": self.experiment_name,
                    "total_runs": len(runs_df),
                    "export_date": datetime.now().isoformat(),
                    "runs_summary": runs_df.describe().to_dict(),
                    "best_runs": {}
                }
                
                # Encontrar melhores execuções para métricas comuns
                metric_columns = [col for col in runs_df.columns if col.startswith('metrics.')]
                for metric_col in metric_columns:
                    metric_name = metric_col.replace('metrics.', '')
                    best_run = self.get_best_run(metric_name)
                    if best_run:
                        report["best_runs"][metric_name] = {
                            "run_id": best_run["run_id"],
                            "value": best_run[metric_col]
                        }
                
                # Salvar relatório
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Relatório exportado: {output_path}")
                
        except Exception as e:
            logger.error(f"Erro ao exportar relatório: {e}")

# Função de conveniência para uso rápido
def quick_experiment(experiment_name: str = "recommendation_system"):
    """
    Criar rapidamente um tracker de experimentos
    
    Args:
        experiment_name: Nome do experimento
        
    Returns:
        Instância do ExperimentTracker
    """
    return ExperimentTracker(experiment_name=experiment_name)

# Exemplo de uso
if __name__ == "__main__":
    # Criar tracker
    tracker = ExperimentTracker()
    
    # Iniciar execução
    run_id = tracker.start_run("test_run")
    
    # Registrar parâmetros de exemplo
    tracker.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })
    
    # Registrar métricas de exemplo
    tracker.log_metrics({
        "mse": 0.001,
        "rmse": 0.032,
        "mae": 0.025
    })
    
    # Finalizar execução
    tracker.end_run()
    
    # Obter relatório
    tracker.export_experiment_report()
    
    print("✓ Teste do ExperimentTracker concluído!")