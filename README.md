# Avaliação de LLMs como Julgadores em QA de Direito Tributário Brasileiro

Pacote de reprodutibilidade do estudo sobre avaliação de modelos de linguagem
(LLMs) em perguntas e respostas (QA) de legislação tributária brasileira,
usando o dataset **Tax Law Brazil COSIT** (101 questões oficiais da Receita
Federal). O experimento compara 11 modelos — abertos e proprietários — por meio
de métricas automáticas (ROUGE-L, BLEU, BERTScore, Connectedness) e do
paradigma *LLM-as-a-judge* com 3 julgadores (GPT, Gemini e Claude, score 0–5),
e analisa a confiabilidade desse julgamento: concordância entre julgadores
(alfa de Krippendorff), viés de circularidade juiz–respondente e alinhamento
entre métricas automáticas e julgamento por LLM.

Autora: Larysa Mendes (UFCG, Campina Grande, PB, Brasil).

## Objetivo

O estudo aborda quatro perguntas de pesquisa:

* **RQ1:** Qual o desempenho de LLMs abertos e proprietários em QA de direito
  tributário brasileiro, segundo julgadores LLM e métricas automáticas?
* **RQ2:** Os 3 julgadores LLM concordam entre si (alfa de Krippendorff,
  unanimidade de categoria)?
* **RQ3:** Existe viés de circularidade — julgadores favorecendo respostas do
  próprio modelo ou família?
* **RQ4:** As métricas automáticas se alinham ao julgamento por LLM, por
  resposta e por modelo?

Em síntese: pelos julgadores, o GPT-5.1 lidera (score médio 4,30/5) e o
ranking por métricas automáticas é quase o inverso do ranking por julgadores; a concordância entre julgadores fica abaixo do limiar de confiabilidade usual (α = 0,628 < 0,667);
e o teste pareado de circularidade mostra o Gemini atribuindo à própria
família +0,41 ponto em relação aos demais julgadores, enquanto Claude e GPT
são mais duros consigo mesmos (−0,41 e −0,21).

## Estrutura do repositório

```
├── src/                              # Código-fonte, organizado por etapa do pipeline
│   ├── 01_geracao_respostas/         #   Geração das respostas dos modelos avaliados
│   │   ├── gerar_respostas_modelos_abertos.py
│   │   ├── gerar_respostas_claude.py
│   │   ├── gerar_respostas_gemini.py
│   │   └── gerar_respostas_gpt.py
│   ├── 02_metricas_automaticas/      #   ROUGE-L, BLEU, BERTScore, Connectedness
│   │   ├── calcular_metricas_automaticas.py
│   │   └── gerar_ranking_metricas.py
│   ├── 03_julgamento_llm/            #   Julgamento das respostas por LLMs juízes
│   │   └── julgar_respostas.py
│   ├── 04_analises/                  #   Krippendorff, circularidade, correlação, erros
│   └── 05_graficos/                  #   Geração de figuras e dashboards
├── data/
│   └── tax_law_brazil_cosit/         # Dataset (perguntas, respostas oficiais, corpus)
├── prompt/                           # Prompts usados no experimento (referência)
│   ├── template_geracao_respostas.txt      # Template de geração (etapa 1, todos os modelos)
│   └── prompt_julgamento_score_0_5.txt     # Prompt dos julgadores, score 0-5 (etapa 3)
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

## Dataset

O dataset **Tax Law Brazil COSIT** está em `data/tax_law_brazil_cosit/`
(formato HuggingFace `datasets`, com exportações em CSV/JSON) e contém 101
perguntas de direito tributário para pessoas jurídicas, extraídas do documento
oficial "Perguntas e Respostas PJ 2023" da COSIT/Receita Federal, mais um
corpus de 30 documentos de apoio.

Cada instância do split `tax_law` tem os campos:

| Campo | Descrição |
|---|---|
| `question` | Pergunta de direito tributário |
| `answer` | Resposta oficial do especialista (COSIT) |
| `reference` | Referência normativa (lei, IN, decreto) |
| `gold_passage` | Passagem-ouro do corpus usada como contexto na geração |

Os pipelines leem o dataset com `datasets.load_from_disk`; não é necessária
nenhuma etapa de conversão. Detalhes adicionais em
`data/tax_law_brazil_cosit/README.md`.

## Modelos avaliados

11 modelos, com os mesmos parâmetros de geração (`temperature = 0.1`,
`max_tokens = 4096`, sem `top_p` explícito, mesmo template de prompt):

* **Proprietários (API própria):** Claude Opus 4.6 · Gemini 3 Pro Preview · GPT-5.1
* **Abertos (via NVIDIA NIM):** DeepSeek V4 Flash · Gemma 4 31B-IT · GPT-OSS 120B ·
  Llama 3.3 70B Instruct · Mixtral 8x22B Instruct v0.1 · Qwen 3.5 122B-A10B ·
  Qwen2 72B Instruct · GLM 5.1

**Julgadores:** GPT-5.1, Gemini 3 Pro Preview e Claude Haiku 4.5. Cada
resposta recebe um score 0–5 de cada julgador; o score oficial é a média dos
scores válidos, com categoria derivada (≥4 CORRETO, ≥3 PARCIALMENTE_CORRETO,
<3 INCORRETO).

## Pipeline

O experimento tem 5 etapas: (1) geração das respostas dos 11 modelos sobre as
101 questões, usando a passagem-ouro como contexto; (2) cálculo das métricas
automáticas de cada resposta contra a resposta oficial; (3) julgamento de cada
resposta pelos 3 LLMs juízes, em lote, com score 0–5 e justificativa em JSON;
(4) análises de concordância, circularidade, correlação e erros; (5) geração
das figuras. As etapas 4 e 5 rodam localmente sobre os CSVs versionados em
`results/`, sem custo de API.

## Prompts

Os prompts estão versionados em `prompt/` e reproduzidos abaixo para inspeção
sem abrir os scripts. Placeholders entre chaves são preenchidos em tempo de
execução.

**Geração de respostas (todos os modelos avaliados).** Placeholders:
`{context}` (passagem-ouro) e `{question}`.

```
Use the following pieces of legal information from laws to answer the user's question.
If the answer is not clear in context, try to figure out by interpreting the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Do not quote the "contextual information" provided in the answer, do not say "according to the information" or anything like that, use the information only to answer the question.
Only return the helpful answer below and nothing else.
REMEMBER: answer the question in portuguese.
Helpful answer:
```

Os scripts do Claude e do GPT usam adicionalmente um system prompt de
assistente jurídico ("Você é um assistente jurídico especialista em direito
tributário brasileiro..."); os scripts do Gemini e dos modelos abertos não
usam system prompt.

**Julgamento (3 julgadores, score 0–5).** Trecho das instruções e da escala; o
prompt completo, incluindo os exemplos few-shot, está em
`prompt/prompt_julgamento_score_0_5.txt`.

```
Instruções:
Avalie a resposta gerada pela IA com base nos seguintes critérios:

1. Verifique se a Resposta da IA está alinhada com a Resposta do Especialista, ou seja, se não existe nenhuma contradição relevante. [...]
2. A Resposta do Especialista pode conter mais informações do que foi solicitado na pergunta. [...]
3. Se a Resposta da IA tiver mais informações do que a Resposta do Especialista, isso não deve ser penalizado automaticamente [...]
4. Verifique se a Resposta da IA responde suficientemente à questão. [...]

A avaliação deve retornar um score escalar de 0 a 5, conforme a seguinte escala:

5 = resposta totalmente correta, suficiente e sem contradições. [...]
4 = resposta correta e suficiente, mas com pequena omissão [...]
3 = resposta parcialmente correta, mas com omissão relevante [...]
2 = resposta contém algum elemento correto, mas erra, omite ponto jurídico importante [...]
1 = resposta majoritariamente incorreta, vaga, confusa [...]
0 = resposta contradiz a Resposta do Especialista, inventa informação essencial [...]

Inclua uma justificativa curta explicando o score atribuído.
A Avaliação deve ser um objeto JSON, com as chaves score e raciocinio.
```

## Como executar

### Instalação

Requer Python 3.10+ (testado com 3.12):

```bash
git clone git@github.com:larysamendes15/Reprodutibilidade.git
cd Reprodutibilidade
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Chaves de API (apenas etapas 1 e 3)

```bash
export OPENAI_API_KEY="sua-chave"
export GEMINI_API_KEY="sua-chave"
export ANTHROPIC_API_KEY="sua-chave"
export NIM_API_KEY="sua-chave"      # NVIDIA NIM (modelos abertos)
```

Nenhum script contém chaves hardcoded; todos abortam com mensagem clara se a
chave faltar. As **etapas 4 e 5 não exigem chaves** e reproduzem todas as
análises e figuras a partir dos CSVs versionados.

### Execução (a partir da raiz do repositório)

```bash
# Etapa 1 — Geração de respostas (requer chaves de API)
python src/01_geracao_respostas/gerar_respostas_modelos_abertos.py   # aceita --max-rows, --resume, --only <alias>
python src/01_geracao_respostas/gerar_respostas_claude.py
python src/01_geracao_respostas/gerar_respostas_gemini.py
python src/01_geracao_respostas/gerar_respostas_gpt.py

# Etapa 2 — Métricas automáticas (baixa modelos do Hugging Face na 1ª execução)
python src/02_metricas_automaticas/calcular_metricas_automaticas.py
python src/02_metricas_automaticas/gerar_ranking_metricas.py

# Etapa 3 — Julgamento por LLMs (requer chaves de API)
python src/03_julgamento_llm/julgar_respostas.py

# Etapa 4 — Análises (sem chaves; executar nesta ordem)
python src/04_analises/analise_krippendorff.py
python src/04_analises/analise_circularidade_juizes.py
python src/04_analises/analise_erros_por_pergunta.py
python src/04_analises/ranking_erros_perguntas.py
python src/04_analises/analise_correlacao_por_resposta.py

# Etapa 5 — Figuras (sem chaves; executar nesta ordem)
python src/05_graficos/graficos_calibracao_e_ranking.py
python src/05_graficos/graficos_explicativos_erros.py --input-dir results/erros --out-dir results/figuras --top-n 14
python src/05_graficos/graficos_top14_erros.py
python src/05_graficos/graficos_dashboard_tcc.py
python src/05_graficos/graficos_score_julgadores.py \
    --summary results/scores_julgadores/summary_101_3_julgadores_score_batch.csv \
    --details results/scores_julgadores/all_results_101_3_julgadores_score_batch.csv \
    --out results/figuras/score_julgadores
```

Notas de reprodução:

* Para rodar `gerar_ranking_metricas.py` **sem** reexecutar a etapa 2, copie o
  agregado versionado antes:
  `cp results/metricas/metrics_all_models_new.csv results/respostas_geradas/`.
* `analise_correlacao_por_resposta.py` usa o cache versionado
  `results/metricas/metricas_por_resposta.csv`; apague-o para recalcular do
  zero (requer PyTorch e download de modelos do Hugging Face).
* `graficos_score_julgadores.py` requer o summary gerado pela etapa 3
  (`summary_101_3_julgadores_score_batch.csv`).
* **Atalho:** para reproduzir apenas as análises e figuras, pule as etapas
  1–3 — as saídas originais estão versionadas em `results/respostas_geradas/`
  e `results/scores_julgadores/`.

## Formato de saída

A etapa 1 gera um `<modelo>_QAG.csv` por modelo em
`results/respostas_geradas/`, com colunas `Q` (pergunta), `A` (resposta
oficial) e `A_model` (resposta do modelo). A etapa 2 acrescenta as métricas
por resposta e o agregado `metrics_all_models_new.csv` (uma linha por modelo).
A etapa 3 gera, em `results/scores_julgadores/`, um CSV por modelo com o score
0–5 e a justificativa de cada julgador, o score oficial (média), a categoria
derivada e o desvio entre julgadores, além do consolidado
`all_results_101_3_julgadores_score_batch.csv` e do resumo por modelo. As
etapas 4 e 5 gravam as tabelas de análise em `results/circularidade/`,
`results/erros/`, `results/ranking_erros_perguntas/` e as figuras em
`results/figuras/` e pastas afins.

## Critérios de avaliação

Cada resposta é avaliada em duas frentes. **Métricas automáticas** contra a
resposta oficial: ROUGE-L (F1), BLEU (sentença, suavizado), BERTScore F1 e
Connectedness (similaridade de cosseno entre embeddings Sentence-BERT
multilíngues). **Julgamento por LLM:** score 0–5 de cada julgador conforme a
escala do prompt; o score oficial é a média dos scores válidos e a categoria
final é CORRETO (≥4), PARCIALMENTE_CORRETO (≥3) ou INCORRETO (<3). Na análise
de erros, uma pergunta conta como errada para um modelo quando a categoria
final não é CORRETO. Seeds fixas (`SEED = 42`) nos scripts de métricas.

## Resultados

As tabelas abaixo foram calculadas a partir dos CSVs versionados em
`results/` (101 questões por modelo, 1.111 respostas julgadas no total).

**RQ1 — Ranking pelos julgadores LLM** (score médio 0–5, média dos 3
julgadores; contagens de categoria final):

| Modelo | Score médio | Correto | Parcial | Incorreto |
|---|---|---|---|---|
| GPT-5.1 | 4,30 | 80 | 14 | 7 |
| Claude Opus 4.6 | 4,17 | 76 | 14 | 11 |
| GLM 5.1 | 4,10 | 72 | 15 | 14 |
| DeepSeek V4 Flash | 4,05 | 70 | 16 | 15 |
| Gemma 4 31B-IT | 4,02 | 73 | 12 | 16 |
| Qwen 3.5 122B-A10B | 4,01 | 69 | 17 | 15 |
| Gemini 3 Pro Preview | 3,99 | 70 | 17 | 14 |
| Mixtral 8x22B Instruct | 3,98 | 68 | 17 | 16 |
| GPT-OSS 120B | 3,87 | 62 | 22 | 17 |
| Qwen2 72B Instruct | 3,85 | 64 | 20 | 17 |
| Llama 3.3 70B Instruct | 3,80 | 67 | 14 | 20 |

**RQ1 — Ranking pelas métricas automáticas** (score geral = média simples das
4 métricas, ×100):

| Modelo | ROUGE-L | BLEU | BERT F1 | Connect. | Score geral |
|---|---|---|---|---|---|
| Mixtral 8x22B Instruct | 0,449 | 0,309 | 0,800 | 0,830 | 59,7 |
| Qwen2 72B Instruct | 0,428 | 0,295 | 0,800 | 0,832 | 58,9 |
| GLM 5.1 | 0,430 | 0,299 | 0,798 | 0,823 | 58,7 |
| Gemma 4 31B-IT | 0,440 | 0,281 | 0,804 | 0,816 | 58,5 |
| Llama 3.3 70B Instruct | 0,425 | 0,313 | 0,793 | 0,801 | 58,3 |
| Qwen 3.5 122B-A10B | 0,426 | 0,258 | 0,797 | 0,825 | 57,6 |
| Gemini 3 Pro Preview | 0,411 | 0,276 | 0,797 | 0,816 | 57,5 |
| DeepSeek V4 Flash | 0,427 | 0,253 | 0,798 | 0,811 | 57,2 |
| GPT-5.1 | 0,362 | 0,153 | 0,774 | 0,825 | 52,8 |
| GPT-OSS 120B | 0,339 | 0,120 | 0,767 | 0,820 | 51,2 |
| Claude Opus 4.6 | 0,302 | 0,096 | 0,733 | 0,784 | 47,8 |

Os dois rankings são quase invertidos: os modelos mais bem avaliados pelos
julgadores (GPT-5.1, Claude Opus 4.6) são os últimos nas métricas de
sobreposição lexical, indicando respostas corretas porém mais parafraseadas em
relação ao texto oficial.

**RQ2 — Concordância entre julgadores.** Alfa de Krippendorff ordinal geral de
**0,628**, com unanimidade de categoria em **70,6%** das 1.111 respostas e
desvio médio de score de 0,476 entre julgadores — abaixo do limiar de 0,667
usualmente aceito para conclusões provisórias, o que recomenda cautela ao usar
um único julgador LLM. Por modelo avaliado, o alfa varia de 0,45 (respostas do
Claude Opus 4.6) a 0,71 (respostas do Qwen2 72B).

**RQ3 — Circularidade juiz–respondente.** Na comparação pareada (mesma
resposta, julgador da mesma família vs. demais julgadores): Gemini atribui à
própria família **+0,41** ponto (IC95% [0,25; 0,57]); Claude atribui **−0,41**
(IC95% [−0,62; −0,21]) e GPT **−0,21** (IC95% [−0,31; −0,11]) — ou seja, no
experimento, apenas o Gemini exibe auto-preferência; Claude e GPT são mais
rigorosos com a própria família.

**RQ4 — Métricas automáticas vs. julgadores.** Por resposta (n = 1.111), as
correlações são moderadas: BERTScore F1 (Pearson 0,50 / Spearman 0,53) e
ROUGE-L (0,48 / 0,53) são as mais alinhadas, seguidas de BLEU (0,38 / 0,45) e
Connectedness (0,37 / 0,34). Por modelo (n = 11), todas as correlações são
negativas e não significativas — as métricas automáticas não reproduzem o
ranking dos julgadores.

## Análise de erros

Os erros se concentram em poucas perguntas: as **5 perguntas mais difíceis
respondem por 33,3%** de todos os erros, as 10 mais difíceis por 58,6% e as
**14 mais difíceis por 72,2%**. A taxa de erro (categoria final INCORRETO)
varia de 6,9% (GPT-5.1) a 19,8% (Llama 3.3 70B). O detalhamento por pergunta,
com as respostas de cada modelo, está em `results/erros/` e
`results/ranking_erros_perguntas/`; o heatmap e a análise de robustez das 14
perguntas mais difíceis estão em `results/figuras/` e
`results/graficos_top14_erros/`.

## Licença

Código sob licença MIT (ver `LICENSE`).

## Como citar

```bibtex
@misc{qa_taxlaw_judges_2026,
  author = {Larysa Mendes},
  title  = {Avaliação de LLMs como Julgadores em QA de Direito Tributário Brasileiro},
  year   = {2026},
  howpublished = {Pacote de reprodutibilidade},
  institution  = {Universidade Federal de Campina Grande (UFCG)},
  url    = {https://github.com/larysamendes15/Reprodutibilidade}
}
```

## Referências

* COSIT/Receita Federal do Brasil. **Perguntas e Respostas Pessoa Jurídica
  2023.** Fonte das 101 questões e respostas oficiais do dataset.
