# Análise de circularidade juiz–respondente
Total de respostas carregadas: **1212**.

## Modelos encontrados por família

### Família: claude
- claude-opus-4-6: 101 respostas

### Família: gemini
- gemini-3-pro-preview: 101 respostas

### Família: gpt
- gpt-5.1: 101 respostas
- openai_gpt-oss-120b: 101 respostas

### Família: outros
- deepseek-ai_deepseek-v4-flash: 101 respostas
- google_gemma-4-31b-it: 101 respostas
- meta_llama-3.3-70b-instruct: 101 respostas
- mistralai_mixtral-8x22b-instruct-v0.1: 101 respostas
- qwen2-72B-Instruct: 202 respostas
- qwen_qwen3.5-122b-a10b: 101 respostas
- z-ai_glm-5.1: 101 respostas

## 1. Juiz avaliando própria família vs demais modelos

Esta análise verifica se cada juiz atribui notas médias maiores aos modelos da sua própria família do que aos demais modelos.

- **GPT**: própria família = 3.941 (n=202), demais modelos = 3.966 (n=1010), Δ = -0.026.
- **GEMINI**: própria família = 4.267 (n=101), demais modelos = 4.160 (n=1111), Δ = 0.107.
- **CLAUDE**: própria família = 3.901 (n=101), demais modelos = 3.864 (n=1111), Δ = 0.037.

## 2. Comparação pareada no mesmo conjunto de respostas

Esta é a análise mais forte. Para cada família de modelo avaliado, compara-se a nota do juiz da mesma família com a média dos outros dois juízes nas mesmas respostas.

- **GPT**: juiz da família = 3.941, outros juízes = 4.153, Δ = -0.213 (IC95 bootstrap: -0.312 a -0.114); sem favorecimento médio.
- **GEMINI**: juiz da família = 4.267, outros juízes = 3.856, Δ = 0.411 (IC95 bootstrap: 0.252 a 0.574); possível favorecimento.
- **CLAUDE**: juiz da família = 3.901, outros juízes = 4.312, Δ = -0.411 (IC95 bootstrap: -0.624 a -0.213); sem favorecimento médio.

## Como reportar no artigo

Sugestão de texto: "Para verificar possível circularidade juiz–respondente, comparamos a nota média atribuída por cada julgador aos modelos de sua própria família com as notas atribuídas aos demais modelos. Além disso, realizamos uma comparação pareada, contrastando, para as mesmas respostas, a nota do julgador da família do modelo com a média dos outros dois julgadores. Essa análise permite verificar se há evidência de favorecimento sistemático associado à família do modelo avaliado."
