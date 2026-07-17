# Avaliação de LLMs como Julgadores em QA de Direito Tributário Brasileiro

Este repositório contém o código, os dados e os resultados de um experimento de
avaliação de modelos de linguagem (LLMs) em tarefas de perguntas e respostas
(QA) sobre legislação tributária brasileira, usando o dataset **Tax Law Brazil
COSIT** (101 questões). O experimento compara respostas de múltiplos modelos por
meio de métricas automáticas (ROUGE-L, BLEU, BERTScore, Connectedness) e do
paradigma *LLM-as-a-judge* (GPT, Gemini e Claude como julgadores), incluindo
análises de concordância entre julgadores (alfa de Krippendorff), viés de
circularidade juiz–respondente, correlação entre métricas e análise de erros
por pergunta.

## Estrutura do repositório

```
├── src/                              # Código-fonte, organizado por etapa do pipeline
│   ├── 01_geracao_respostas/         #   Geração das respostas dos modelos avaliados
│   │   ├── gerar_respostas_modelos_abertos.py
│   │   ├── gerar_respostas_claude.py
│   │   └── gerar_respostas_gemini.py
│   ├── 02_metricas_automaticas/      #   ROUGE-L, BLEU, BERTScore, Connectedness
│   │   ├── calcular_metricas_automaticas.py
│   │   └── gerar_ranking_metricas.py
│   ├── 03_julgamento_llm/            #   Julgamento das respostas por LLMs juízes
│   │   └── julgar_respostas.py
│   ├── 04_analises/                  #   Krippendorff, circularidade, correlação, erros
│   ├── 05_graficos/                  #   Geração de figuras e dashboards
│   └── documentacao/
│       └── pipeline.md               #   Descrição detalhada de cada script
├── data/
│   └── tax_law_brazil_cosit/         # Dataset (perguntas, respostas oficiais, corpus)
├── results/
│   ├── respostas_geradas/            # Respostas geradas por modelo (*_QAG.csv)
│   ├── scores_julgadores/            # Scores atribuídos pelos 3 LLMs julgadores
│   ├── metricas/                     # Métricas automáticas, rankings e cache por resposta
│   ├── circularidade/                # Análise de viés juiz–respondente (self vs. outros)
│   ├── erros/                        # Análise de erros por modelo e por pergunta
│   └── figuras/                      # Gráficos finais (PNG/PDF)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

Algumas pastas de saída (`results/ranking_erros_perguntas/`,
`results/graficos_top14_erros/`, `results/graficos_tcc_final/`) são criadas
automaticamente pelos scripts das etapas 4 e 5.

## Modelos avaliados

Claude Opus 4.6 · DeepSeek V4 Flash · Gemini 3 Pro Preview · Gemma 4 31B-IT ·
GPT-5.1 · GPT-OSS 120B · Llama 3.3 70B Instruct · Mixtral 8x22B Instruct v0.1 ·
Qwen 3.5 122B-A10B · Qwen2 72B Instruct · GLM 5.1

**Julgadores:** GPT, Gemini e Claude (3 julgadores por resposta, 101 questões,
score de 0 a 5).

## Requisitos

- Python 3.10+ (testado com Python 3.12)
- Dependências em `requirements.txt`. PyTorch e sentence-transformers são
  necessários apenas para BERTScore/Connectedness (etapa 2); GPU é recomendada,
  mas não obrigatória. Na primeira execução da etapa 2, os modelos de embedding
  são baixados do Hugging Face (requer internet).

## Instalação

```bash
git clone git@github.com:larysamendes15/Reprodutibilidade.git
cd Reprodutibilidade
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Chaves de API

As etapas 1 (geração de respostas) e 3 (julgamento por LLMs) chamam APIs
externas. Defina as chaves como **variáveis de ambiente** — nenhum script
contém chaves hardcoded e todos abortam com mensagem clara se a chave faltar:

```bash
export OPENAI_API_KEY="sua-chave"
export GEMINI_API_KEY="sua-chave"
export ANTHROPIC_API_KEY="sua-chave"
export NIM_API_KEY="sua-chave"      # NVIDIA NIM (modelos abertos)
```

As etapas 4 e 5 **não exigem chaves**: rodam localmente sobre os CSVs já
versionados em `results/`, o que permite reproduzir todas as análises e figuras
sem custo de API.

## Como reproduzir o experimento

Todos os scripts devem ser executados **a partir da raiz do repositório**
(os caminhos internos são relativos a ela).

### Etapa 1 — Geração de respostas (requer chaves de API)

```bash
python src/01_geracao_respostas/gerar_respostas_modelos_abertos.py   # modelos abertos via NVIDIA NIM
python src/01_geracao_respostas/gerar_respostas_claude.py            # Claude (Anthropic)
python src/01_geracao_respostas/gerar_respostas_gemini.py            # Gemini (Google)
```

O gerador de modelos abertos aceita `--max-rows N` (teste rápido), `--resume`
(continuar de checkpoint) e `--only <alias>` (rodar um único modelo). Saída:
`results/respostas_geradas/<modelo>_QAG.csv`.

### Etapa 2 — Métricas automáticas

```bash
python src/02_metricas_automaticas/calcular_metricas_automaticas.py
python src/02_metricas_automaticas/gerar_ranking_metricas.py
```

O primeiro script gera `results/respostas_geradas/metrics_all_models_new.csv`
(e um `<modelo>_metrics.csv` por modelo); o segundo lê esse arquivo e produz os
rankings em `results/metricas/`.

> **Nota:** o agregado versionado no repositório está em
> `results/metricas/metrics_all_models_new.csv`. Para rodar
> `gerar_ranking_metricas.py` **sem** reexecutar a etapa 2, copie-o antes:
>
> ```bash
> cp results/metricas/metrics_all_models_new.csv results/respostas_geradas/
> ```

### Etapa 3 — Julgamento por LLMs (requer chaves de API)

```bash
python src/03_julgamento_llm/julgar_respostas.py
```

Submete cada resposta de `results/respostas_geradas/*_QAG.csv` aos 3
julgadores e gera os CSVs de `results/scores_julgadores/`, incluindo o
consolidado `all_results_101_3_julgadores_score_batch.csv` e o resumo por
modelo `summary_101_3_julgadores_score_batch.csv` (este último não está
versionado; é necessário para `graficos_score_julgadores.py`).

### Etapa 4 — Análises (sem chaves de API)

Execute nesta ordem — `ranking_erros_perguntas.py` depende da saída de
`analise_erros_por_pergunta.py`:

```bash
python src/04_analises/analise_krippendorff.py             # concordância entre julgadores (stdout)
python src/04_analises/analise_circularidade_juizes.py     # viés juiz–respondente -> results/circularidade/
python src/04_analises/analise_erros_por_pergunta.py       # erros por modelo/pergunta -> results/erros/
python src/04_analises/ranking_erros_perguntas.py          # -> results/ranking_erros_perguntas/
python src/04_analises/analise_correlacao_por_resposta.py  # correlação métricas x julgadores
```

O script de correlação usa o cache versionado
`results/metricas/metricas_por_resposta.csv`. Apague o cache para recalcular as
métricas do zero (requer PyTorch e download de modelos do Hugging Face).

### Etapa 5 — Figuras (sem chaves de API)

Execute nesta ordem — o dashboard consome saídas dos gráficos anteriores:

```bash
python src/05_graficos/graficos_calibracao_e_ranking.py    # -> results/figuras/*.pdf
python src/05_graficos/graficos_explicativos_erros.py --input-dir results/erros --out-dir results/figuras --top-n 14
python src/05_graficos/graficos_top14_erros.py             # -> results/graficos_top14_erros/
python src/05_graficos/graficos_dashboard_tcc.py           # -> results/graficos_tcc_final/
python src/05_graficos/graficos_score_julgadores.py \
    --summary results/scores_julgadores/summary_101_3_julgadores_score_batch.csv \
    --details results/scores_julgadores/all_results_101_3_julgadores_score_batch.csv \
    --out results/figuras/score_julgadores
```

> **Atalho:** quem quiser apenas reproduzir as análises e figuras pode pular as
> etapas 1–3 — as saídas originais já estão versionadas em
> `results/respostas_geradas/` e `results/scores_julgadores/`.

A descrição completa de cada script (entradas, saídas e função) está em
[`src/documentacao/pipeline.md`](src/documentacao/pipeline.md).

## Dados

O dataset está em `data/tax_law_brazil_cosit/` (formato HuggingFace
`datasets`, com exportações em CSV/JSON). São 101 perguntas de direito
tributário para pessoas jurídicas extraídas do documento "Perguntas e
Respostas PJ 2023" da COSIT/Receita Federal, com resposta oficial, referência
normativa e passagem-ouro, além de um corpus de 30 documentos de apoio.
Detalhes em `data/tax_law_brazil_cosit/README.md`.

## Reprodutibilidade

- Seeds fixas (`SEED = 42`) nos scripts de métricas.
- As etapas 4 e 5 são determinísticas sobre os CSVs versionados.
- As etapas 1 e 3 dependem de APIs de terceiros e podem produzir respostas
  ligeiramente diferentes entre execuções, mesmo com temperatura baixa.

## Licença

Código sob licença MIT (ver `LICENSE`).

## Como citar

```bibtex
@misc{qa_taxlaw_judges_2026,
  author = {Larysa Mendes},
  title  = {Avaliação de LLMs como Julgadores em QA de Direito Tributário Brasileiro},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXXXX},
  url    = {https://github.com/larysamendes15/Reprodutibilidade}
}
```
