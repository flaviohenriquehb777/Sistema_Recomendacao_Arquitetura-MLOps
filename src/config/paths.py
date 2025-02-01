from pathlib import Path
import sys

def get_project_root() -> Path:
    """Retorna a raiz do projeto baseado em marcadores."""
    current = Path(__file__).resolve()
    while not any((current / marker).exists() for marker in [".git", "src", "dados"]):
        if current.parent == current:
            raise RuntimeError("Raiz do projeto não encontrada.")
        current = current.parent
    return current


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "dados"
DADOS_BRUTOS = DATA_DIR / "criando_uma_apresentação_executiva.csv"  # Nome padronizado
DADOS_TRATADOS = DATA_DIR / "dados_tratados.parquet"  # Exemplo

if not DADOS_BRUTOS.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {DADOS_BRUTOS}")


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)  # Cria a pasta se não existir


REPORTS_DIR = PROJECT_ROOT / "report"
REPORTS_DIR.mkdir(exist_ok=True) 