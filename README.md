# Avaliação de LLMs como Julgadores em QA de Direito Tributário Brasileiro

Este repositório contém o código, os dados e os resultados de um experimento de
avaliação de modelos de linguagem (LLMs) em tarefas de perguntas e respostas
(QA) sobre legislação tributária brasileira, usando o dataset **Tax Law Brazil
COSIT**. O experimento compara respostas de múltiplos modelos por meio de
métricas automáticas (ROUGE, BLEU, BERTScore, Connectedness) e do paradigma
*LLM-as-a-judge* (GPT, Gemini e Claude como julgadores), incluindo análises de
concordância entre julgadores (alfa de Krippendorff), viés de circularidade
juiz–respondente, correlação entre métricas e análise de erros por pergunta.

## Estrutura do repositório

```
├── src/                              # Código-fonte, organizado por etapa do pipeline
│   ├── 01_geracao_respostas/         #   Geração das respostas dos modelos avaliados
│   ├── 02_metricas_automaticas/      #   ROUGE, BLEU, BERTScore, Connectedness
│   ├── 03_julgamento_llm/            #   Julgamento das respostas por LLMs juízes
│   ├── 04_analises/                  #   Krippendorff, circularidade, correlação, erros
│   └── 05_graficos/                  #   Geração de figuras e dashboards
├── data/
│   └── tax_law_brazil_cosit/         # Dataset (perguntas, respostas oficiais, corpus)
├── results/
│   ├── qag/                          # Respostas geradas + métricas automáticas por modelo
│   ├── judges_scores/                # Scores atribuídos pelos 3 LLMs julgadores
│   ├── circularidade/                # Análise de viés juiz–respondente (self vs. outros)
│   ├── erros/                        # Análise de erros por modelo e por pergunta
│   ├── metricas/                     # Rankings agregados e métricas por resposta
│   └── figuras/                      # Gráficos finais (PNG/PDF)
├── docs/
│   └── pipeline.md                   # Descrição detalhada de cada script
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Modelos avaliados

Claude Opus 4.6 · DeepSeek V4 Flash · Gemini 3 Pro Preview · Gemma 4 31B-IT ·
GPT-5.1 · GPT-OSS 120B · Llama 3.3 70B Instruct · Mixtral 8x22B Instruct v0.1 ·
Qwen 3.5 122B-A10B · Qwen2 72B Instruct · GLM 5.1

**Julgadores:** GPT, Gemini e Claude (3 julgadores por resposta, 101 questões).

## Requisitos

- Python 3.10+
- Dependências em `requirements.txt` (inclui PyTorch e sentence-transformers
  para o BERTScore/Connectedness; GPU é recomendada, mas não obrigatória)

## Instalação

```bash
git clone git@github.com:larysamendes15/Reprodutibilidade.git
cd SEU_REPO
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Chaves de API

As etapas 1 (geração de respostas) e 3 (julgamento por LLMs) chamam APIs
externas. Defina as chaves como **variáveis de ambiente** — nenhum script
contém chaves hardcoded:

```bash
export OPENAI_API_KEY="sua-chave"
export GEMINI_API_KEY="sua-chave"
export ANTHROPIC_API_KEY="sua-chave"
export NIM_API_KEY="sua-chave"      # NVIDIA NIM (modelos abertos)
```

As etapas 2, 4 e 5 **não exigem chaves**: rodam localmente sobre os CSVs já
presentes em `results/`, o que permite reproduzir todas as análises e figuras
sem custo de API.

## Como reproduzir o experimento

Todos os scripts devem ser executados **a partir da raiz do repositório**
(os caminhos internos são relativos a ela).

```bash
# Etapa 1 — Gerar respostas dos modelos (requer chaves de API)
python src/01_geracao_respostas/qa_nim_multi_models.py   # modelos abertos via NVIDIA NIM
python src/01_geracao_respostas/qa_claude.py             # Claude
python src/01_geracao_respostas/qa_gemini.py             # Gemini

# Etapa 2 — Métricas automáticas (ROUGE, BLEU, BERTScore, Connectedness)
python src/02_metricas_automaticas/eval_metrics_all_models_v2.py

# Etapa 3 — Julgamento das respostas por LLMs juízes (requer chaves de API)
python src/03_julgamento_llm/score_julgadores.py

# Etapa 4 — Análises
python src/04_analises/analise_krippendorff.py           # concordância entre julgadores
python src/04_analises/analise_circularidade_juizes.py   # viés juiz–respondente
python src/04_analises/analise_correlacao_por_resposta.py
python src/04_analises/analisa_erros_por_pergunta.py
python src/04_analises/ranking_erros_perguntas.py

# Etapa 5 — Figuras
python src/05_graficos/gerar_figuras_2_e_4.py
python src/05_graficos/graficos_dashboard_tcc.py
```

> **Atalho:** quem quiser apenas reproduzir as análises e figuras pode pular
> as etapas 1–3 — as saídas originais já estão versionadas em
> `results/qag/` e `results/judges_scores/`.

A descrição completa de cada script (entradas, saídas e função) está em
[`docs/pipeline.md`](docs/pipeline.md).

## Dados

O dataset está em `data/tax_law_brazil_cosit/` (formato HuggingFace
`datasets`, com exportações em CSV/JSON). São perguntas de direito tributário
para pessoas jurídicas extraídas do documento "Perguntas e Respostas PJ 2023"
da COSIT/Receita Federal, com resposta oficial, referência normativa e
passagem-ouro. Detalhes em `data/tax_law_brazil_cosit/README.md`.

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
