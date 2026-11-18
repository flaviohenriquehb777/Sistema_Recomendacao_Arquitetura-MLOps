"""
Módulo para configuração e utilização do MLflow para rastreamento de experimentos.

O tracking é OPCIONAL e NÃO-BLOQUEANTE:
- Se a variável de ambiente ``RECOMENDACAO_MLFLOW_TRACKING_URI`` estiver definida
  (ex.: servidor remoto do DagsHub deste projeto), ela é usada.
- Caso contrário, usa o backend LOCAL em SQLite (``sqlite:///<raiz>/mlflow.db``),
  o backend recomendado pelo MLflow 3 — o antigo file store (``mlruns/``) entrou
  em "maintenance mode" e não recebe novas features.
- As variáveis do projeto são carregadas do arquivo ``.env`` (python-dotenv).
  Usamos uma variável PRÓPRIA (``RECOMENDACAO_MLFLOW_TRACKING_URI``) para não
  herdar o ``MLFLOW_TRACKING_URI`` global, que pode apontar para outro projeto.
- Qualquer falha de conexão/registro é avisada e ignorada, sem interromper o pipeline.
"""
import os
import re
from pathlib import Path
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from dotenv import load_dotenv

from src.config.paths import PROJECT_ROOT


def _sanitize_name(name: str) -> str:
    """MLflow 3 só aceita nomes com alfanuméricos, ``_ - . /`` e espaço."""
    return re.sub(r"[^A-Za-z0-9_\-./ ]", "_", str(name))

# Carrega as variáveis do arquivo .env do projeto (se existir) antes de lê-las.
load_dotenv(str(PROJECT_ROOT / ".env"))

# Diretório local de artefatos legado (file store), mantido só por compatibilidade.
MLRUNS_DIR = Path(PROJECT_ROOT) / "mlruns"

# Banco SQLite local: backend recomendado pelo MLflow 3. O arquivo fica na raiz do
# projeto e é ignorado pelo Git.
MLFLOW_DB_PATH = Path(PROJECT_ROOT) / "mlflow.db"

# Nome do experimento MLflow deste projeto.
MLFLOW_EXPERIMENT_NAME = os.getenv("RECOMENDACAO_MLFLOW_EXPERIMENT", "recomendacao_produtos")

# URI de rastreamento (opt-in remoto). Usamos uma variável PRÓPRIA do projeto para
# não herdar o MLFLOW_TRACKING_URI global, que pode apontar para outro repositório
# (ex.: outro projeto DagsHub). Para usar um servidor remoto:
#   export RECOMENDACAO_MLFLOW_TRACKING_URI=https://dagshub.com/<owner>/<repo>.mlflow
# Padrão: backend local SQLite, permitindo treinar 100% offline.
MLFLOW_TRACKING_URI = os.getenv(
    "RECOMENDACAO_MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

def setup_mlflow(experiment_name: str | None = None):
    """
    Configura o MLflow com o URI de rastreamento.

    Não levanta exceção: em caso de falha, avisa e retorna None para que o
    treino prossiga sem tracking.

    Args:
        experiment_name: nome do experimento a ativar. Se None, usa o
            experimento padrão do projeto (``RECOMENDACAO_MLFLOW_EXPERIMENT``).

    Returns:
        str | None: URI efetiva de tracking em caso de sucesso; None se falhar.
    """
    nome_experimento = experiment_name or MLFLOW_EXPERIMENT_NAME
    try:
        if MLFLOW_TRACKING_URI.startswith("file:"):
            MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        elif MLFLOW_TRACKING_URI.startswith("http"):
            # Tracking remoto (ex.: DagsHub): autentica com o token DESTE projeto,
            # sobrescrevendo qualquer MLFLOW_TRACKING_* global (de outro projeto).
            token = os.getenv("DAGSHUB_USER_TOKEN")
            if token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
                    "DAGSHUB_REPO_OWNER", "flaviohenriquehb777"
                )
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        # Garante que o experimento exista e esteja ativo (evita "experiment ID 0").
        if mlflow.get_experiment_by_name(nome_experimento) is None:
            mlflow.create_experiment(nome_experimento)
        mlflow.set_experiment(nome_experimento)
        tracking_uri = mlflow.get_tracking_uri()
        print(f"MLflow configurado com URI: {tracking_uri} | Experimento: {nome_experimento}")
        return tracking_uri
    except Exception as e:
        print(f"[aviso] Falha ao configurar o MLflow ({e}). Continuando sem tracking.")
        return None

def log_run(params: dict, metrics: dict, artifacts=None, run_name="run"):
    """
    Registra **parâmetros**, **métricas** e (opcionalmente) **artefatos** de um
    run — SEM registrar o modelo no Model Registry.

    Indicada para estágios que não precisam versionar o modelo neural (ex.: o
    benchmark de escala sobre a base sintética).

    Não-bloqueante: qualquer falha (servidor indisponível, valor inválido, etc.)
    é avisada e ignorada, sem interromper o pipeline.

    Args:
        params: dicionário de parâmetros/metadados (valores escalares).
        metrics: dicionário de métricas numéricas.
        artifacts: iterável de caminhos de arquivos a anexar ao run.
        run_name: nome do run.

    Returns:
        str | None: run ID em caso de sucesso; None se o tracking falhar.
    """
    try:
        with mlflow.start_run(run_name=run_name) as run:
            for chave, valor in (params or {}).items():
                if isinstance(valor, (str, int, float, bool)):
                    mlflow.log_param(_sanitize_name(chave), valor)
                else:
                    mlflow.log_param(_sanitize_name(chave), str(valor))
            for chave, valor in (metrics or {}).items():
                mlflow.log_metric(_sanitize_name(chave), float(valor))
            for caminho in (artifacts or []):
                if Path(caminho).exists():
                    mlflow.log_artifact(str(caminho))
            print(f"Run '{run_name}' registrado no MLflow. Run ID: {run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        print(f"[aviso] Falha ao registrar o run no MLflow ({e}). Continuando sem tracking.")
        return None


def log_model_metrics(model, X_test, y_test, metrics_dict, model_name="recommendation_model"):
    """
    Registra métricas e o modelo no MLflow.

    Não-bloqueante: se o tracking falhar (servidor indisponível, modelo não
    serializável, etc.), avisa e retorna None, sem interromper o treino.

    Args:
        model: Modelo treinado (TensorFlow/Keras)
        X_test: Dados de teste (features)
        y_test: Dados de teste (target)
        metrics_dict: Dicionário com métricas de avaliação
        model_name: Nome do modelo para registro

    Returns:
        str | None: Run ID em caso de sucesso; None se o tracking falhar.
    """
    try:
        with mlflow.start_run(run_name=model_name) as run:
            # Registrar parâmetros do modelo
            for layer in model.layers:
                if hasattr(layer, 'get_config'):
                    config = layer.get_config()
                    for key, value in config.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"{layer.name}_{key}", value)

            # Registrar métricas (nome saneado: "ndcg@7" -> "ndcg_7").
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(_sanitize_name(metric_name), metric_value)

            # Assinatura do modelo (best-effort). O flavor Keras do MLflow 3 exige
            # assinaturas com TensorSpec para modelos multi-input; para não falhar o
            # registro, só inferimos a assinatura quando a entrada é única.
            try:
                if isinstance(X_test, (list, tuple)):
                    signature = None
                else:
                    signature = infer_signature(X_test, model.predict(X_test))
            except Exception as e:
                print(f"[aviso] Assinatura do modelo indisponível ({e}). Registrando sem assinatura.")
                signature = None

            # Salvar o modelo com o flavor Keras nativo do MLflow 3.
            mlflow.keras.log_model(
                model,
                name="model",
                signature=signature,
                registered_model_name=model_name,
            )

            print(f"Modelo registrado com sucesso. Run ID: {run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        print(f"[aviso] Falha ao registrar no MLflow ({e}). Continuando sem tracking.")
        return None

def load_registered_model(model_name="recommendation_model", alias="champion"):
    """
    Carrega um modelo registrado do MLflow por alias.

    O MLflow 3 removeu os "stages" (Production/Staging) em favor de **aliases**,
    então o padrão passa a ser ``champion``.

    Args:
        model_name: Nome do modelo registrado.
        alias: Alias da versão (ex.: ``champion``).

    Returns:
        Modelo carregado, ou None em caso de falha.
    """
    model_uri = f"models:/{model_name}@{alias}"
    try:
        model = mlflow.keras.load_model(model_uri)
        print(f"Modelo {model_name} (alias: {alias}) carregado com sucesso.")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

if __name__ == "__main__":
    # Teste da configuração
    tracking_uri = setup_mlflow()
    print(f"MLflow Tracking URI: {tracking_uri}")