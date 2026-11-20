from pathlib import Path


def get_project_root() -> Path:
    """Retorna a raiz do projeto baseado em marcadores."""
    current = Path(__file__).resolve()
    while not any((current / marker).exists() for marker in [".git", "src", "dados"]):
        if current.parent == current:
            raise RuntimeError("Raiz do projeto não encontrada.")
        current = current.parent
    return current


PROJECT_ROOT = get_project_root()

# Uma pasta por base, identificada apenas por nome-código. O vínculo entre o
# código e a fonte real (e as licenças de cada base) fica fora do repositório.
DATA_DIR = PROJECT_ROOT / "dados"
SC_BASE_01_DIR = DATA_DIR / "sc_base_01"
SC_BASE_03_DIR = DATA_DIR / "sc_base_03"
SC_BASE_04_DIR = DATA_DIR / "sc_base_04"
SC_BASE_05_DIR = DATA_DIR / "sc_base_05"

DADOS_BRUTOS = SC_BASE_01_DIR / "sc_base_01_bruto.csv"
DADOS_TRATADOS = SC_BASE_01_DIR / "sc_base_01_tratado.parquet"


def exigir_dados_brutos() -> Path:
    """Caminho do CSV bruto de `sc_base_01`, validado apenas no momento do uso.

    A validação vive aqui — e não no import do módulo — para que `paths` possa
    ser importado em ambientes sem os dados (CI, clone limpo), onde o bruto é
    armazenado fora do git.
    """
    if not DADOS_BRUTOS.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DADOS_BRUTOS}")
    return DADOS_BRUTOS


MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)  # Cria a pasta se não existir


REPORTS_DIR = PROJECT_ROOT / "report"
REPORTS_DIR.mkdir(exist_ok=True)
