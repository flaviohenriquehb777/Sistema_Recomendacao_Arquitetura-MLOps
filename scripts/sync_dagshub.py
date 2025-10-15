#!/usr/bin/env python3
"""
Script para sincronização com DagsHub
"""

import subprocess
import sys
import os
import json
from datetime import datetime

def run_command(command, description):
    """Executar comando e mostrar resultado"""
    print(f"\n🔄 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - Sucesso")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} - Erro")
            if result.stderr.strip():
                print(f"Erro: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao executar comando: {e}")
        return False
    
    return True

def check_dvc_status():
    """Verificar status do DVC"""
    print("\n📊 Verificando status do DVC...")
    
    commands = [
        ("dvc status", "Verificar status dos dados"),
        ("dvc remote list -v", "Verificar remotes configurados"),
        ("dvc dag", "Verificar pipeline de dados")
    ]
    
    for command, description in commands:
        run_command(command, description)

def check_dagshub_config():
    """Verificar configuração do DagsHub"""
    print("\n⚙️ Verificando configuração do DagsHub...")
    
    # Verificar arquivo de configuração DVC
    dvc_config_path = ".dvc/config"
    if os.path.exists(dvc_config_path):
        print(f"✅ Arquivo de configuração DVC encontrado: {dvc_config_path}")
        
        with open(dvc_config_path, 'r') as f:
            config_content = f.read()
            print("Configuração atual:")
            print(config_content)
    else:
        print("❌ Arquivo de configuração DVC não encontrado")
        return False
    
    # Verificar se o remote dagshub está configurado
    result = subprocess.run("dvc remote list", shell=True, capture_output=True, text=True)
    if "dagshub" in result.stdout:
        print("✅ Remote DagsHub configurado")
    else:
        print("❌ Remote DagsHub não configurado")
        return False
    
    return True

def setup_dagshub_remote():
    """Configurar remote do DagsHub se necessário"""
    print("\n⚙️ Configurando remote do DagsHub...")
    
    dagshub_url = "https://dagshub.com/flaviohenriquehb777/Projeto_7_Sistema_de_Recomendacao.dvc"
    
    commands = [
        (f"dvc remote add -d dagshub {dagshub_url}", "Adicionar remote DagsHub"),
        ("dvc remote modify dagshub user flaviohenriquehb777", "Configurar usuário"),
    ]
    
    for command, description in commands:
        run_command(command, description)
    
    print("\n⚠️ IMPORTANTE: Configure o token do DagsHub:")
    print("dvc remote modify dagshub password SEU_DAGSHUB_TOKEN")
    print("\nPara obter o token:")
    print("1. Acesse https://dagshub.com/user/settings/tokens")
    print("2. Crie um novo token")
    print("3. Execute: dvc remote modify dagshub password SEU_TOKEN")

def sync_data_to_dagshub():
    """Sincronizar dados com DagsHub"""
    print("\n🚀 Iniciando sincronização de dados com DagsHub...")
    
    # Verificar se DVC está inicializado
    if not os.path.exists('.dvc'):
        print("❌ DVC não está inicializado. Execute 'dvc init' primeiro.")
        return False
    
    # Verificar configuração
    if not check_dagshub_config():
        print("⚠️ Configuração do DagsHub incompleta. Configurando...")
        setup_dagshub_remote()
        return False
    
    # Verificar status
    check_dvc_status()
    
    # Comandos de sincronização
    commands = [
        ("dvc add data/", "Adicionar dados ao DVC (se necessário)"),
        ("dvc push", "Enviar dados para DagsHub")
    ]
    
    success = True
    for command, description in commands:
        if not run_command(command, description):
            if "dvc add" in command:
                print("⚠️ Dados já podem estar sendo rastreados pelo DVC")
                continue
            else:
                success = False
                break
    
    if success:
        print("\n✅ Sincronização de dados com DagsHub concluída!")
    else:
        print("\n❌ Erro na sincronização com DagsHub")
        print("\n🔧 Possíveis soluções:")
        print("1. Verificar se o token está configurado corretamente")
        print("2. Verificar conectividade com a internet")
        print("3. Verificar se o repositório DagsHub existe")
    
    return success

def sync_experiments():
    """Sincronizar experimentos MLflow"""
    print("\n🧪 Sincronizando experimentos MLflow...")
    
    try:
        # Importar módulo de tracking
        sys.path.append('src')
        from config.experiment_tracker import ExperimentTracker
        
        # Criar tracker
        tracker = ExperimentTracker()
        
        # Exportar relatório de experimentos
        tracker.export_experiment_report("reports/experiment_report.json")
        
        print("✅ Relatório de experimentos exportado")
        
    except ImportError as e:
        print(f"⚠️ Não foi possível importar ExperimentTracker: {e}")
    except Exception as e:
        print(f"❌ Erro ao sincronizar experimentos: {e}")

def create_dvc_pipeline():
    """Criar pipeline DVC se não existir"""
    print("\n📋 Verificando pipeline DVC...")
    
    dvc_yaml_path = "dvc.yaml"
    
    if not os.path.exists(dvc_yaml_path):
        print("📝 Criando pipeline DVC...")
        
        pipeline_content = """stages:
  data_preparation:
    cmd: python src/scripts/otimizacao_modelo_simples.py
    deps:
    - src/scripts/otimizacao_modelo_simples.py
    - data/raw/
    outs:
    - data/processed/
    
  model_training:
    cmd: python src/scripts/otimizacao_modelo.py
    deps:
    - src/scripts/otimizacao_modelo.py
    - data/processed/
    outs:
    - models/optimized/
    metrics:
    - metrics/model_metrics.json
    
  ensemble_training:
    cmd: python src/scripts/modelo_ensemble.py
    deps:
    - src/scripts/modelo_ensemble.py
    - models/optimized/
    outs:
    - models/ensemble/
    metrics:
    - metrics/ensemble_metrics.json
"""
        
        with open(dvc_yaml_path, 'w', encoding='utf-8') as f:
            f.write(pipeline_content)
        
        print("✅ Pipeline DVC criado")
    else:
        print("✅ Pipeline DVC já existe")

def check_environment():
    """Verificar ambiente e dependências"""
    print("\n🔍 Verificando ambiente...")
    
    # Verificar se DVC está instalado
    result = subprocess.run("dvc version", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ DVC instalado: {result.stdout.strip()}")
    else:
        print("❌ DVC não está instalado. Execute: pip install dvc[all]")
        return False
    
    # Verificar se dagshub está instalado
    try:
        import dagshub
        print(f"✅ DagsHub instalado: {dagshub.__version__}")
    except ImportError:
        print("❌ DagsHub não está instalado. Execute: pip install dagshub")
        return False
    
    # Verificar se mlflow está instalado
    try:
        import mlflow
        print(f"✅ MLflow instalado: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow não está instalado. Execute: pip install mlflow")
        return False
    
    return True

def main():
    """Função principal"""
    print("🔄 Script de Sincronização com DagsHub")
    print("=" * 50)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("src") or not os.path.exists("requirements.txt"):
        print("⚠️ Execute este script na raiz do projeto")
        return
    
    # Verificar ambiente
    if not check_environment():
        return
    
    # Criar pipeline se necessário
    create_dvc_pipeline()
    
    # Sincronizar dados
    if sync_data_to_dagshub():
        # Sincronizar experimentos
        sync_experiments()
    
    print("\n📋 Comandos úteis para sincronização manual:")
    print("# Dados:")
    print("dvc add data/")
    print("dvc push")
    print("dvc pull")
    print("\n# Status:")
    print("dvc status")
    print("dvc dag")
    print("\n# Pipeline:")
    print("dvc repro")
    print("dvc metrics show")
    
    print("\n🔗 Links úteis:")
    print("DagsHub: https://dagshub.com/flaviohenriquehb777/Projeto_7_Sistema_de_Recomendacao")
    print("MLflow: https://dagshub.com/flaviohenriquehb777/Projeto_7_Sistema_de_Recomendacao.mlflow")

if __name__ == "__main__":
    main()