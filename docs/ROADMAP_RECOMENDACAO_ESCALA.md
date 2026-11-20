# Roadmap: Recomendação Personalizada em Escala

**Início previsto:** 21/11/2026
**Objetivo:** transformar o projeto em uma vitrine técnica de recomendação personalizada, reproduzível e preparada para servir modelos, sem alegar equivalência com sistemas comerciais da Amazon ou do Mercado Livre.

## Política de nomenclatura das bases

- Todas as bases são identificadas apenas por código: `sc_base_01`, `sc_base_02`, `sc_base_03`.
- Nenhum nome de origem, fornecedor ou competição aparece em código, nomes de arquivo/pasta, dados, notebooks, pipeline DVC, Docker ou README.
- O vínculo código → fonte real e as licenças ficam registrados na memória do projeto, fora do repositório.
- Dados brutos nunca são versionados no Git, enviados ao DVC remoto ou incluídos na imagem Docker.
- A obtenção das bases é uma etapa manual e externa: nenhuma plataforma de dados, credencial ou ferramenta de download é dependência do projeto.

## Estratégia de dados

- **`sc_base_01`** — base real de varejo transacional; baseline didático e referência para demonstrar que o protocolo não força personalização quando ela não é sustentada pela validação.
- **`sc_base_02`** — benchmark principal de escala: compras por cliente com atributos de artigo e de cliente. Uso restrito a fins não comerciais e pesquisa acadêmica; exige aceite das regras da competição na conta do usuário antes do download. Não publicar os dados, não incluí-los na imagem Docker e não enviar dados, pesos ou artefatos derivados para um remoto DVC sem confirmar que os termos permitem.
- **`sc_base_03`** — benchmark comportamental complementar: visualização, carrinho e compra ao longo de meses. Licença com cláusulas de uso não comercial e compartilhamento pela mesma licença; manter atribuição e respeitar as restrições de redistribuição.
- **Bases de avaliação textual em escala** — fora do escopo inicial: avaliações não são registros completos de exposição ou compra, a escala aumenta bastante o custo e os direitos de uso precisam ser esclarecidos antes de qualquer publicação.

Para uso comercial real, treinar e validar com dados próprios do cliente, obtidos com autorização. Bases públicas restritas servem como benchmarks de portfólio, não como dados de produção comercial.

## Etapa transversal obrigatória — Entendimento e tratamento de dados

Aplica-se a **todas** as bases, antes de qualquer modelagem.

- **Entendimento:** volume, cobertura temporal, cardinalidade de usuários/itens, tipos, memória, granularidade do evento e significado de cada campo.
- **Qualidade:** valores ausentes, duplicatas exatas e lógicas, identificadores órfãos, categorias inconsistentes, outliers, formatação e codificação de texto.
- **Viés e representatividade:** concentração de interações em cauda longa, usuários e itens com pouquíssimas observações, período coberto, sazonalidade, sobreposição entre treino e teste e risco de vazamento temporal.
- **Ruído:** distinguir ruído de sinal legítimo antes de descartar qualquer registro. Ausência de interação não é rejeição comprovada.
- **Tratamento:** regras explícitas e reproduzíveis, com registro do que foi removido, imputado ou normalizado e do impacto esperado.
- **Saída:** relatório reprodutível de entendimento/qualidade + estágio DVC de preparação por base.

## Etapas

### 0. Alinhamento e guardrails

- Confirmar acesso aos arquivos de `sc_base_02` e reler os termos vigentes antes do download.
- Definir armazenamento local ignorado pelo Git; manter dados brutos fora da imagem Docker.
- Decidir quais métricas e artefatos podem ser publicados conforme os termos aplicáveis a cada base.
- Manter um registro explícito das licenças, fontes e limitações.

**Saída:** decisão documentada de uso permitido e layout de dados sem risco de inclusão acidental no Git, DVC remoto ou container.

### 1. Entendimento e tratamento de `sc_base_01`

- Rodar o profiling completo e registrar as limitações conhecidas (densidade baixa, ausência de sinal de exposição).
- Remover `sc_base_01` do papel de modelo principal da API e mantê-lo como baseline.

**Saída:** relatório de qualidade da base e decisão registrada sobre seu uso como baseline.

**Status:** concluída (20/11/2026). Relatório reprodutível em `report/perfil_sc_base_01.json`, produzido pelo estágio DVC `perfil_dados` — 9.800 linhas, 793 clientes, 1.861 produtos, esparsidade 0,9934, top 10 % dos produtos concentrando 19,96 % das interações.

**Decisão:** `sc_base_01` é o **baseline** do projeto, não o modelo principal da API. O serviço é agnóstico de base — `DATA_PATH` declara o parquet servido e `MODEL_VARIANT` declara o papel do conjunto (hoje `baseline`, exposto em `GET /health`). O modelo principal só será definido quando existir treino em `sc_base_02`/`sc_base_03` (Etapas 2–4); até lá o `alpha = 0,0` selecionado por validação mantém o modo `popularity` nesta base.

**Higiene de dados:** o CSV bruto de `sc_base_01` deixou de ser versionado (passou a `dados/.gitignore`), cumprindo a política de dados brutos fora do Git. A validação de existência foi movida do import para o uso, para que o código seja importável em clone limpo.

### 2. Ingestão, entendimento e tratamento de `sc_base_02`

- Ler transações, clientes e artigos sem descartar data, IDs ou atributos relevantes.
- Auditar volume, cobertura temporal, clientes/artigos sem metadados, duplicatas, valores ausentes, formatação de texto e compras repetidas.
- Definir o evento-alvo e a política para artigos já comprados.
- Criar relatório de qualidade reproduzível, sem expor dados restritos.

**Saída:** estágio de preparação DVC reproduzível e relatório com contagens, qualidade e limitações.

### 3. Protocolo temporal e baselines

- Usar separação cronológica alinhada à tarefa de prever a semana seguinte; evitar divisão aleatória e vazamento temporal.
- Reservar períodos distintos para treino, validação e teste final.
- Definir elegibilidade de usuários warm-start e avaliar usuários novos separadamente.
- Implementar baselines de popularidade, recência e coocorrência item-item.

**Saída:** split congelado e baselines medidos no mesmo horizonte e catálogo candidato.

### 4. Modelos personalizados e recuperação de candidatos

- Começar com recomendação implícita por fatoração ou coocorrência, comparada aos baselines.
- Separar a arquitetura em duas etapas: recuperação de candidatos por usuário e ranqueamento dos candidatos.
- Incorporar atributos de artigo e sinais de recência/frequência apenas quando disponíveis no instante da recomendação.
- Tratar amostras negativas como não observadas, não como rejeições comprovadas; documentar a hipótese usada.
- Escolher o modelo pela validação, sem impor peso personalizado positivo.

**Saída:** candidatos personalizados reproduzíveis e uma configuração selecionada somente se apresentar ganho validado.

### 5. Avaliação e decisão do modelo

- Reportar NDCG@12, Recall@12, MAP/Hit Rate, cobertura de catálogo e métricas de diversidade/novidade.
- Desagregar resultados por atividade do cliente, popularidade do artigo e cold start.
- Comparar com todos os baselines no teste temporal intocado e calcular incerteza por bootstrap.
- Rejeitar ou revisar o modelo se o ganho não for consistente; não substituir resultado negativo por alpha forçado.

**Critério de saída:** melhoria demonstrável sobre o melhor baseline, sem vazamento e sem degradação inaceitável de cobertura. Se não houver melhoria, registrar a conclusão e investigar dados/objetivo.

### 6. Artefatos de produção via DVC

- Criar estágio de treinamento específico para `sc_base_02`, separado do estágio de `sc_base_01`.
- Versionar código, configuração, métricas permitidas e metadados do modelo; manter dados e artefatos restritos locais até confirmar autorização para compartilhamento.
- Persistir modelo de recuperação, modelo de ranking, encoders/catálogo e identificadores de versão compatíveis entre si.
- Registrar data de treino, janela temporal, features, licença e métricas associadas.

**Saída:** artefatos versionados e reproduzíveis que a API possa carregar, sem mistura de modelos ou encoders de bases diferentes.

### 7. Serving, Docker e CI

- Adaptar FastAPI para carregar os componentes de recuperação e ranking uma vez no startup.
- Preservar `/health` e `/recommend`, informar versão e modo de ranking e manter fallback explícito para cold start.
- Testar contrato, validação, recomendações, explicação do score realmente servido e carregamento de artefatos.
- Construir e executar a imagem Docker com os artefatos autorizados; não incluir CSVs brutos.
- Rodar pytest, Ruff e Black no escopo e smoke tests de `/health` e `/recommend`.

**Saída:** container reproduzível e demonstrado com o modelo personalizado selecionado, não apenas com o baseline de popularidade.

### 8. Entendimento e tratamento de `sc_base_03`

- Preservar tipo e timestamp do evento na ingestão.
- Comparar experimentos com visualização, carrinho e transação separados ou ponderados; medir compras como alvo à parte.
- Reavaliar o benchmark temporalmente e documentar que o resultado existente é de um recorte filtrado de usuários engajados.
- Manter `sc_base_03` como evidência complementar, sem misturar seus encoders ou artefatos com `sc_base_02`.

**Saída:** comparação independente em sinais comportamentais e narrativa clara sobre generalização e limites.

### 9. Vitrine e model card

- Atualizar README com arquitetura, protocolo, métricas reais, licença, reprodutibilidade e limitações.
- Publicar instruções e scripts, não dados brutos nem artefatos cuja redistribuição não esteja autorizada.
- Distinguir claramente “preparado para integração” de “implantado e validado em produção”.

**Saída:** demonstração honesta e auditável para entrevistas e avaliação técnica por empresas.

## Estado de partida

- `sc_base_01`: a validação escolheu `alpha=0`; o modelo dessa base permanece baseline.
- `sc_base_03`: o benchmark existente reporta NDCG@7 de 0,0541 contra 0,0117 para popularidade, no recorte documentado.
- `sc_base_02`: ainda não está no projeto; depende de aceite das regras e de credenciais do usuário.
- Correções locais recentes incluem normalização de espaço não separável, SHAP sobre o score efetivo de ranking e `ranking_mode` explícito.
- A suíte local passou com **26 testes**; `ruff check .` e `black --check` (escopo do CI) também estão limpos. O rebuild Docker posterior a essas correções foi interrompido e continua pendente.
- A nomenclatura anônima das bases foi concluída e o histórico do Git foi reescrito, de modo que nenhum nome real de origem permanece nos commits, nos caminhos ou no conteúdo versionado.
- Exposição residual conhecida: a ref `refs/pull/1/head` mantida pelo GitHub a partir de uma PR antiga ainda aponta para um commit de antes da reescrita.
