#!/usr/bin/env python3
"""
Script para verificar se os experimentos estão aparecendo no DagsHub
"""

import sys
import mlflow
import requests
from datetime import datetime

# Adicionar src ao path
sys.path.append('src')
from config.dagshub_config import setup_dagshub_mlflow

def verify_dagshub_experiments():
    """
    Verifica se os experimentos estão aparecendo no DagsHub
    """
    
    print("🔍 Verificando experimentos no DagsHub...")
    
    # Configurar MLflow
    setup_dagshub_mlflow("verificacao_experimentos")
    
    try:
        # Listar todos os experimentos
        experiments = mlflow.search_experiments()
        
        print(f"\n📊 Total de experimentos encontrados: {len(experiments)}")
        print("=" * 60)
        
        for exp in experiments:
            print(f"🧪 Experimento: {exp.name}")
            print(f"   📝 ID: {exp.experiment_id}")
            print(f"   📅 Criado: {datetime.fromtimestamp(exp.creation_time/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   🔄 Status: {exp.lifecycle_stage}")
            
            # Buscar runs deste experimento
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"   🏃 Runs: {len(runs)}")
            
            if len(runs) > 0:
                print("   📈 Últimas métricas:")
                for idx, run in runs.head(3).iterrows():
                    print(f"      • Run: {run['tags.mlflow.runName'] if 'tags.mlflow.runName' in run else 'N/A'}")
                    
                    # Mostrar algumas métricas principais
                    metrics_to_show = ['mse', 'precision_at_5', 'final_test_mse', 'final_test_mae']
                    for metric in metrics_to_show:
                        if f'metrics.{metric}' in run:
                            print(f"         {metric}: {run[f'metrics.{metric}']:.4f}")
            
            print("-" * 40)
        
        # Verificar conectividade com DagsHub
        print("\n🌐 Verificando conectividade com DagsHub...")
        
        dagshub_url = "https://dagshub.com/flaviohenriquehb777/Projeto_7_Sistema_de_Recomendacao"
        experiments_url = f"{dagshub_url}/experiments"
        
        try:
            response = requests.get(dagshub_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ DagsHub acessível: {dagshub_url}")
                print(f"🔗 Link dos experimentos: {experiments_url}")
            else:
                print(f"⚠️  DagsHub retornou status {response.status_code}")
        except requests.RequestException as e:
            print(f"❌ Erro ao acessar DagsHub: {e}")
        
        # Informações do tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        print(f"\n📡 MLflow Tracking URI: {tracking_uri}")
        
        # Resumo final
        print("\n" + "=" * 60)
        print("📋 RESUMO DA VERIFICAÇÃO")
        print("=" * 60)
        print(f"✅ Experimentos configurados: {len(experiments)}")
        
        total_runs = sum(len(mlflow.search_runs(experiment_ids=[exp.experiment_id])) for exp in experiments)
        print(f"✅ Total de runs executados: {total_runs}")
        
        print(f"✅ Tracking URI configurado: {tracking_uri}")
        print(f"🔗 Acesse os experimentos em: {experiments_url}")
        
        if total_runs > 0:
            print("\n🎉 SUCESSO! Os experimentos foram criados e devem estar visíveis no DagsHub.")
            print("💡 Se não estão aparecendo na interface web, pode levar alguns minutos para sincronizar.")
        else:
            print("\n⚠️  Nenhum run encontrado. Execute os scripts de experimentos primeiro.")
            
    except Exception as e:
        print(f"❌ Erro durante verificação: {e}")
        import traceback
        traceback.print_exc()

def show_experiment_links():
    """
    Mostra links diretos para os experimentos
    """
    
    print("\n🔗 LINKS DIRETOS PARA VERIFICAÇÃO:")
    print("=" * 50)
    
    base_url = "https://dagshub.com/flaviohenriquehb777/Projeto_7_Sistema_de_Recomendacao"
    
    links = [
        ("🏠 Página principal do projeto", base_url),
        ("🧪 Experimentos MLflow", f"{base_url}/experiments"),
        ("📊 Métricas e comparações", f"{base_url}/experiments/compare"),
        ("📈 Dashboard MLflow", f"{base_url}.mlflow"),
        ("🔧 Configurações do projeto", f"{base_url}/settings")
    ]
    
    for description, url in links:
        print(f"{description}: {url}")

if __name__ == "__main__":
    print("🔍 Iniciando verificação dos experimentos no DagsHub...")
    
    try:
        verify_dagshub_experiments()
        show_experiment_links()
        
        print("\n✅ Verificação concluída!")
        
    except Exception as e:
        print(f"❌ Erro durante verificação: {e}")
        import traceback
        traceback.print_exc()