from pathlib import Path

# 1. Encontra a raiz do projeto subindo até encontrar um marcador (como .git, ou pasta src)
def encontrar_raiz_projeto():
    caminho = Path(__file__).resolve()
    while not (caminho / ".git").exists() and not (caminho / "src").exists() and caminho.parent != caminho:
        caminho = caminho.parent
    return caminho

# 2. Define os caminhos
PASTA_PROJETO = encontrar_raiz_projeto()
PASTA_DADOS = PASTA_PROJETO / "Dados"
DADOS_BRUTOS = PASTA_DADOS / "Criando uma apresentação executiva.csv"
DADOS_TRATADOS = PASTA_DADOS / "Criando uma apresentação executiva.parquet"

# 3. Verificação com mensagem mais informativa
if not DADOS_BRUTOS.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {DADOS_BRUTOS}\n"
        f"Diretório atual: {Path.cwd()}\n"
        f"Verifique se você está executando do diretório correto:\n"
        f"Raiz do projeto esperada: {PASTA_PROJETO}"
    )


