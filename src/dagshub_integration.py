"""
Módulo para integração com DagsHub.
Facilita o uso do DagsHub para rastreamento de experimentos MLflow e versionamento DVC.
"""

import os
import mlflow
from dotenv import load_dotenv

# Carrega variáveis de ambiente de um arquivo .env (se existir)
load_dotenv()

def setup_dagshub_tracking(username=None, repo_name=None):
    """
    Configura o MLflow para usar o DagsHub como servidor de rastreamento.
    
    Args:
        username (str, optional): Nome de usuário do DagsHub. Se None, tenta obter de variáveis de ambiente.
        repo_name (str, optional): Nome do repositório no DagsHub. Se None, tenta obter de variáveis de ambiente.
    
    Returns:
        str: URI de rastreamento do MLflow configurado
    """
    # Tenta obter credenciais de variáveis de ambiente se não fornecidas
    username = username or os.getenv("DAGSHUB_USERNAME")
    repo_name = repo_name or os.getenv("DAGSHUB_REPO_NAME", "Projeto_7_Sistema_de_Recomendacao")
    
    if not username:
        print("AVISO: Nome de usuário do DagsHub não fornecido. Usando configuração local do MLflow.")
        return None
    
    # Configura o URI de rastreamento do MLflow para o DagsHub
    tracking_uri = f"https://dagshub.com/{username}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow configurado para usar o DagsHub: {tracking_uri}")
    return tracking_uri

def setup_dagshub_credentials(token=None):
    """
    Configura as credenciais do DagsHub para acesso à API.
    
    Args:
        token (str, optional): Token de acesso do DagsHub. Se None, tenta obter de variáveis de ambiente.
    """
    token = token or os.getenv("DAGSHUB_TOKEN")
    
    if not token:
        print("AVISO: Token do DagsHub não fornecido. Algumas operações podem falhar.")
        return
    
    # Configura variáveis de ambiente para autenticação
    os.environ["MLFLOW_TRACKING_USERNAME"] = "token"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    print("Credenciais do DagsHub configuradas com sucesso.")

def push_dvc_to_dagshub():
    """
    Executa o comando DVC push para enviar dados e modelos para o remote configurado.
    Requer que o DVC já esteja configurado com um remote.
    
    Returns:
        bool: True se o push foi bem-sucedido, False caso contrário.
    """
    import subprocess
    
    try:
        result = subprocess.run(["dvc", "push"], check=True, capture_output=True, text=True)
        print("DVC push concluído com sucesso.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar DVC push: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False

def pull_dvc_from_dagshub():
    """
    Executa o comando DVC pull para baixar dados e modelos do remote configurado.
    Requer que o DVC já esteja configurado com um remote.
    
    Returns:
        bool: True se o pull foi bem-sucedido, False caso contrário.
    """
    import subprocess
    
    try:
        result = subprocess.run(["dvc", "pull"], check=True, capture_output=True, text=True)
        print("DVC pull concluído com sucesso.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar DVC pull: {e}")
        print(f"Saída de erro: {e.stderr}")
        return False

if __name__ == "__main__":
    # Exemplo de uso
    setup_dagshub_tracking()
    setup_dagshub_credentials()
    
    # Exemplo de como usar em um fluxo de trabalho
    print("\nExemplo de fluxo de trabalho com DagsHub:")
    print("1. Configure o MLflow para usar o DagsHub")
    print("2. Execute experimentos com MLflow")
    print("3. Faça push dos dados e modelos versionados com DVC")