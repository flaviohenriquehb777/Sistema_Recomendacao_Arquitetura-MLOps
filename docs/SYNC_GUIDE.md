# 🔄 Guia de Sincronização - GitHub e DagsHub

Este guia fornece instruções completas para sincronizar seu projeto com GitHub (código) e DagsHub (dados e experimentos).

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Sincronização com GitHub](#sincronização-com-github)
3. [Sincronização com DagsHub](#sincronização-com-dagshub)
4. [Experimentos MLflow](#experimentos-mlflow)
5. [Comandos Rápidos](#comandos-rápidos)
6. [Solução de Problemas](#solução-de-problemas)

## 🔧 Pré-requisitos

### Ferramentas Necessárias
```bash
# Instalar dependências
pip install dvc[all] dagshub mlflow

# Verificar instalações
git --version
dvc version
python -c "import dagshub; print(dagshub.__version__)"
python -c "import mlflow; print(mlflow.__version__)"
```

### Configuração Inicial do Git
```bash
# Configurar usuário (se ainda não configurado)
git config --global user.name "Seu Nome"
git config --global user.email "seu.email@exemplo.com"

# Verificar configuração
git config --global --list
```

## 🐙 Sincronização com GitHub

### Método 1: Script Automatizado
```bash
# Executar script de sincronização
python scripts/sync_github.py
```

### Método 2: Comandos Manuais
```bash
# 1. Verificar status
git status

# 2. Adicionar arquivos
git add .

# 3. Fazer commit
git commit -m "Atualização do projeto - $(date)"

# 4. Enviar para GitHub
git push origin main
# ou se usar branch master:
git push origin master
```

### Configuração Inicial do Repositório
```bash
# Se ainda não é um repositório Git
git init

# Adicionar remote do GitHub
git remote add origin https://github.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.git

# Verificar remotes
git remote -v
```

## 🎯 Sincronização com DagsHub

### Método 1: Script Automatizado
```bash
# Executar script de sincronização
python scripts/sync_dagshub.py
```

### Método 2: Configuração Manual

#### 1. Inicializar DVC (se necessário)
```bash
# Inicializar DVC
dvc init

# Verificar se foi inicializado
ls -la .dvc/
```

#### 2. Configurar Remote DagsHub
```bash
# Adicionar remote DagsHub
dvc remote add -d dagshub https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.dvc

# Configurar usuário
dvc remote modify dagshub user flaviohenriquehb777

# Configurar token (OBRIGATÓRIO)
dvc remote modify dagshub password SEU_DAGSHUB_TOKEN
```

#### 3. Obter Token do DagsHub
1. Acesse: https://dagshub.com/user/settings/tokens
2. Clique em "New Token"
3. Dê um nome ao token (ex: "projeto-recomendacao")
4. Selecione as permissões necessárias
5. Copie o token gerado
6. Execute: `dvc remote modify dagshub password SEU_TOKEN_AQUI`

#### 4. Adicionar e Sincronizar Dados
```bash
# Adicionar dados ao DVC
dvc add data/

# Fazer commit dos arquivos .dvc
git add data.dvc .dvcignore
git commit -m "Adicionar dados ao DVC"

# Enviar dados para DagsHub
dvc push

# Enviar código para GitHub
git push origin main
```

## 🧪 Experimentos MLflow

### Configurar Tracking
```python
# No seu código Python
import sys
sys.path.append('src')
from config.experiment_tracker import ExperimentTracker

# Criar tracker
tracker = ExperimentTracker()

# Iniciar experimento
run_id = tracker.start_run("meu_experimento")

# Registrar parâmetros
tracker.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
})

# Registrar métricas
tracker.log_metrics({
    "mse": 0.001,
    "rmse": 0.032,
    "mae": 0.025
})

# Finalizar
tracker.end_run()
```

### Visualizar Experimentos
- **MLflow UI Local**: `mlflow ui`
- **DagsHub MLflow**: https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.mlflow

## ⚡ Comandos Rápidos

### Sincronização Completa
```bash
# 1. Sincronizar código com GitHub
python scripts/sync_github.py

# 2. Sincronizar dados com DagsHub
python scripts/sync_dagshub.py

# 3. Verificar status
git status
dvc status
```

### Comandos de Verificação
```bash
# Git
git log --oneline -5
git remote -v
git branch -a

# DVC
dvc status
dvc remote list -v
dvc dag

# MLflow
mlflow experiments list
```

### Pipeline DVC
```bash
# Executar pipeline completo
dvc repro

# Ver métricas
dvc metrics show

# Comparar experimentos
dvc metrics diff
```

## 🔧 Solução de Problemas

### Problema: "Authentication failed" no DagsHub
**Solução:**
```bash
# Verificar se o token está configurado
dvc remote list -v

# Reconfigurar token
dvc remote modify dagshub password SEU_NOVO_TOKEN

# Testar conexão
dvc push --remote dagshub
```

### Problema: "No such file or directory" no Git
**Solução:**
```bash
# Verificar se está no diretório correto
pwd
ls -la

# Inicializar repositório se necessário
git init
git remote add origin https://github.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.git
```

### Problema: "DVC is not initialized"
**Solução:**
```bash
# Inicializar DVC
dvc init

# Verificar inicialização
ls -la .dvc/
```

### Problema: Conflitos de Merge
**Solução:**
```bash
# Fazer pull antes do push
git pull origin main

# Resolver conflitos manualmente
# Depois fazer commit e push
git add .
git commit -m "Resolver conflitos"
git push origin main
```

## 📊 Estrutura de Arquivos Sincronizados

### GitHub (Código)
```
├── src/                    # Código fonte
├── notebooks/              # Jupyter notebooks
├── tests/                  # Testes unitários
├── .github/workflows/      # GitHub Actions
├── requirements.txt        # Dependências
├── README.md              # Documentação
└── scripts/               # Scripts de sincronização
```

### DagsHub (Dados)
```
├── data/                  # Dados (via DVC)
├── models/                # Modelos treinados
├── metrics/               # Métricas dos modelos
└── mlruns/               # Experimentos MLflow
```

## 🎯 Fluxo de Trabalho Recomendado

1. **Desenvolvimento Local**
   ```bash
   # Fazer alterações no código
   # Treinar modelos
   # Executar testes
   ```

2. **Sincronização**
   ```bash
   # Sincronizar código
   python scripts/sync_github.py
   
   # Sincronizar dados
   python scripts/sync_dagshub.py
   ```

3. **Verificação**
   ```bash
   # Verificar GitHub Actions
   # Verificar experimentos no DagsHub
   # Validar métricas
   ```

## 🔗 Links Úteis

- **Repositório GitHub**: https://github.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps
- **Projeto DagsHub**: https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps
- **MLflow UI**: https://dagshub.com/flaviohenriquehb777/Sistema_Recomendacao_Arquitetura-MLOps.mlflow
- **Documentação DVC**: https://dvc.org/doc
- **Documentação DagsHub**: https://dagshub.com/docs

---

💡 **Dica**: Execute os scripts de sincronização regularmente para manter tudo atualizado!

🆘 **Suporte**: Se encontrar problemas, verifique os logs dos comandos e consulte a seção de solução de problemas.
