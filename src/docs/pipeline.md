# Pipeline do experimento — descrição dos scripts

Todos os scripts devem ser executados a partir da **raiz do repositório**.

## Etapa 1 — Geração de respostas (`src/01_geracao_respostas/`)

| Script | Função | Saída |
|---|---|---|
| `qa_nim_multi_models.py` | Gera respostas para todos os modelos abertos via NVIDIA NIM (Llama, Mixtral, Qwen, Gemma, DeepSeek, GLM, GPT-OSS...) com checkpoints | `results/*_QAG.csv` |
| `qa_claude.py` | Gera respostas com o Claude (API Anthropic) | `results/legacy/` |
| `qa_gemini.py` | Gera respostas com o Gemini (API Google) | `results/legacy/` |
| `qa_nim_multi.py` | Versão anterior do gerador NIM (mantida por histórico) | `results/legacy/` |

Requerem: `NIM_API_KEY`, `ANTHROPIC_API_KEY` ou `GEMINI_API_KEY` conforme o script.

## Etapa 2 — Métricas automáticas (`src/02_metricas_automaticas/`)

| Script | Função |
|---|---|
| `eval_metrics_all_models_v2.py` | **Versão atual.** Calcula ROUGE, BLEU, BERTScore e Connectedness para todos os `*_QAG.csv` em `results/qag/`, gerando `<modelo>_metrics.csv` e o agregado `metrics_all_models_new.csv` |
| `eval_metrics_all_models.py` | Versão anterior (lia `results/legacy/`), mantida por histórico |
| `avaliar_metrics_all_models.py` | Gera rankings agregados (`results/metricas/ranking_geral_metricas.csv`, `melhor_modelo_por_metrica.csv`) |
| `eval_claude_4-5.py`, `eval_gemini_2-5.py`, `eval_gemini_3.py`, `eval_gpt_5.py` | Avaliações individuais por modelo usadas em rodadas anteriores do experimento |

Não requerem chaves de API (rodam localmente; baixam modelos do HuggingFace na primeira execução).

## Etapa 3 — Julgamento por LLMs (`src/03_julgamento_llm/`)

| Script | Função |
|---|---|
| `score_julgadores.py` | **Versão atual.** Submete cada resposta em `results/qag/*_QAG.csv` a 3 julgadores (GPT, Gemini e Claude), com suporte a batch, gerando os CSVs de `results/judges_scores/` |
| `mult_judge.py` | Versão anterior do julgamento múltiplo, mantida por histórico |

Requerem: `OPENAI_API_KEY`, `GEMINI_API_KEY` e `ANTHROPIC_API_KEY`.

## Etapa 4 — Análises (`src/04_analises/`)

| Script | Função | Saída |
|---|---|---|
| `analise_krippendorff.py` | Alfa de Krippendorff (concordância entre os 3 julgadores) | stdout |
| `analise_circularidade_juizes.py` | Viés de auto-preferência: score dado pelo juiz ao próprio modelo/família vs. outros, com gráficos e teste pareado | `results/circularidade/` |
| `analise_correlacao_por_resposta.py` | Correlação entre score dos julgadores e métricas automáticas, por resposta (usa cache `results/metricas/metricas_por_resposta.csv`) | stdout + cache |
| `analisa_erros_por_pergunta.py` | Classifica erros por modelo e pergunta | `results/erros/` |
| `ranking_erros_perguntas.py` | Ranking das perguntas mais erradas, com respostas | `results/ranking_erros_perguntas/` |
| `analisa_summary_julgadores.py` | Análises do resumo de scores (rodada anterior, `results/legacy/`) | conforme `--out-dir` |

## Etapa 5 — Gráficos (`src/05_graficos/`)

| Script | Função | Saída |
|---|---|---|
| `gerar_figuras_2_e_4.py` | Figuras 2 e 4 do trabalho (a partir dos scores dos julgadores) | conforme configuração |
| `graficos_explicativos_erros.py` | Gráficos explicativos da análise de erros | `results/figuras/` |
| `graficos_top14_erros.py` | Heatmap/robustez das 14 perguntas mais difíceis | `results/graficos_top14_erros/` |
| `graficos_dashboard_tcc.py` | Dashboard consolidado (modelos × perguntas × Pareto) | `results/graficos_tcc_final/` |
| `graficos_score_julgadores.py` | Gráficos dos scores por julgador + tabela extra | conforme `--out-dir` |
| `gera_graficos_erros.py` | Versão anterior dos gráficos de erros | conforme `--out-dir` |

## Observações

- Os arquivos em `results/legacy/` referem-se a caminhos usados em rodadas
  anteriores do experimento; os scripts marcados como "versão anterior" foram
  mantidos para rastreabilidade dos resultados históricos.
- `results/qag/qwen2-72B-Instruct_old_metrics.csv` é a única versão de
  métricas disponível para o Qwen2 72B (rodada antiga), mantida por isso.
