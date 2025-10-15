#!/usr/bin/env python3
"""
Script para sincronização com GitHub
"""

import subprocess
import sys
import os
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

def check_git_status():
    """Verificar status do git"""
    print("\n📊 Verificando status do repositório...")
    
    commands = [
        ("git status --porcelain", "Verificar arquivos modificados"),
        ("git branch --show-current", "Verificar branch atual"),
        ("git remote -v", "Verificar remotes configurados")
    ]
    
    for command, description in commands:
        run_command(command, description)

def sync_to_github():
    """Sincronizar código com GitHub"""
    print("\n🚀 Iniciando sincronização com GitHub...")
    
    # Verificar se estamos em um repositório git
    if not os.path.exists('.git'):
        print("❌ Não é um repositório Git. Execute 'git init' primeiro.")
        return False
    
    # Verificar status
    check_git_status()
    
    # Comandos de sincronização
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Atualização automática - {timestamp}"
    
    commands = [
        ("git add .", "Adicionar todos os arquivos"),
        (f'git commit -m "{commit_message}"', "Fazer commit das alterações"),
        ("git push origin main", "Enviar para GitHub (branch main)")
    ]
    
    # Tentar push para main, se falhar tentar master
    success = True
    for command, description in commands[:-1]:  # Executar add e commit primeiro
        if not run_command(command, description):
            success = False
            break
    
    if success:
        # Tentar push para main
        if not run_command("git push origin main", "Enviar para GitHub (branch main)"):
            # Se falhar, tentar master
            print("⚠️ Push para 'main' falhou, tentando 'master'...")
            if not run_command("git push origin master", "Enviar para GitHub (branch master)"):
                success = False
    
    if success:
        print("\n✅ Sincronização com GitHub concluída com sucesso!")
    else:
        print("\n❌ Erro na sincronização com GitHub")
    
    return success

def setup_git_if_needed():
    """Configurar git se necessário"""
    print("\n⚙️ Verificando configuração do Git...")
    
    # Verificar se git está configurado
    result = subprocess.run("git config --global user.name", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("⚠️ Git não está configurado. Configure com:")
        print("git config --global user.name 'Seu Nome'")
        print("git config --global user.email 'seu.email@exemplo.com'")
        return False
    
    print(f"✅ Git configurado para: {result.stdout.strip()}")
    return True

def create_gitignore_if_needed():
    """Criar .gitignore se não existir"""
    gitignore_path = ".gitignore"
    
    if not os.path.exists(gitignore_path):
        print("\n📝 Criando .gitignore...")
        
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
models/
*.h5
*.pkl
*.joblib
mlruns/
.dvc/cache/
data/raw/
data/processed/
logs/
*.log

# MLflow
mlruns/
mlartifacts/

# DVC
.dvc/cache/
.dvc/tmp/
"""
        
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        
        print("✅ .gitignore criado")
    else:
        print("✅ .gitignore já existe")

def main():
    """Função principal"""
    print("🔄 Script de Sincronização com GitHub")
    print("=" * 50)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("src") or not os.path.exists("requirements.txt"):
        print("⚠️ Execute este script na raiz do projeto")
        return
    
    # Configurações iniciais
    if not setup_git_if_needed():
        return
    
    create_gitignore_if_needed()
    
    # Sincronizar
    sync_to_github()
    
    print("\n📋 Comandos úteis para sincronização manual:")
    print("git add .")
    print("git commit -m 'Sua mensagem de commit'")
    print("git push origin main")
    print("\n📋 Para verificar status:")
    print("git status")
    print("git log --oneline -10")

if __name__ == "__main__":
    main()