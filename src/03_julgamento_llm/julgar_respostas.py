"""
score_julgadores.py

Avaliação automática de respostas de LLM usando GPT, Gemini e Claude como julgadores.

Diferença para o script binário:
- Em vez de CORRETO / INCORRETO, cada julgador retorna um score escalar de 0 a 5.
- O score oficial é a média dos scores válidos dos julgadores.
- Também gera uma categoria derivada:
    score >= 4   -> CORRETO
    score >= 3   -> PARCIALMENTE_CORRETO
    score < 3    -> INCORRETO

Entrada:
    results/respostas_geradas/*_QAG.csv

Saída:
    results/scores_julgadores/

Colunas esperadas no CSV:
    Q, A, A_model

Instalar:
    pip install openai google-genai anthropic pandas tqdm numpy
"""

from pathlib import Path
from collections import Counter
import json
import re
import time
import os
import random
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from openai import OpenAI
from google import genai
from google.genai import types
import anthropic


# =========================
# CHAVES DAS APIS
# =========================
# Recomendo usar variável de ambiente, para não deixar chave no código.
#
# No terminal:
# export OPENAI_API_KEY="sua-chave"
# export GEMINI_API_KEY="sua-chave"
# export ANTHROPIC_API_KEY="sua-chave"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# =========================
# CONFIGURAÇÕES
# =========================

CSV_DIR = Path("results/respostas_geradas")
OUTPUT_DIR = Path("results/scores_julgadores")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUESTOES_TESTE = 101

JUDGE_BATCH_SIZE = 20

REQUIRED_COLS = ["Q", "A", "A_model"]

OUTPUT_SUFFIX = "_julgadores_score.csv"
PROMPT_VERSION = "prompt"

SLEEP_BETWEEN_BATCHES = 5.0
MAX_RETRIES = 5

JUDGES = {
    "gpt": {
        "provider": "openai",
        "model": "gpt-5.1",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-3-pro-preview",
    },
    "claude": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    },
}

SCORE_MIN = 0.0
SCORE_MAX = 5.0

SYSTEM_PROMPT = (
    "Você é um avaliador jurídico especialista em direito tributário brasileiro. "
    "Siga rigorosamente as instruções do usuário e retorne apenas um JSON válido."
)

DISABLED_JUDGES = set()


# =========================
# PROMPT COM SCORE 0 A 5
# =========================

prompt_eval_score = """
Instruções:
Avalie a resposta gerada pela IA com base nos seguintes critérios:

1. Verifique se a Resposta da IA está alinhada com a Resposta do Especialista, ou seja, se não existe nenhuma contradição relevante. Ignore termos diferentes ou pequenas informações a mais ou a menos, desde que não mudem o sentido jurídico da resposta.

2. A Resposta do Especialista pode conter mais informações do que foi solicitado na pergunta. Se uma informação da Resposta do Especialista não for necessária para responder à pergunta, não use essa informação para penalizar a Resposta da IA.

3. Se a Resposta da IA tiver mais informações do que a Resposta do Especialista, isso não deve ser penalizado automaticamente, desde que as informações adicionais estejam corretas, não gerem confusão jurídica e não contradigam a Resposta do Especialista.

4. Verifique se a Resposta da IA responde suficientemente à questão. Para isso, veja se ela fornece as informações solicitadas na pergunta. Se a questão puder ser respondida com um simples "Não" ou "Sim", isso é aceitável, desde que a resposta esteja juridicamente correta.

A avaliação deve retornar um score escalar de 0 a 5, conforme a seguinte escala:

5 = resposta totalmente correta, suficiente e sem contradições. Informações adicionais corretas não devem reduzir o score, desde que não confundam nem mudem o sentido jurídico da resposta.

4 = resposta correta e suficiente, mas com pequena omissão, menor precisão ou excesso de informação desnecessária que não compromete juridicamente a resposta.

3 = resposta parcialmente correta, mas com omissão relevante ou excesso de informação que torna a resposta menos clara, menos objetiva ou pode gerar dúvida sobre o ponto jurídico principal.

2 = resposta contém algum elemento correto, mas erra, omite ponto jurídico importante ou adiciona informação extra juridicamente problemática.

1 = resposta majoritariamente incorreta, vaga, confusa ou pouco alinhada à Resposta do Especialista.

0 = resposta contradiz a Resposta do Especialista, inventa informação essencial ou responde de forma juridicamente errada.

Inclua uma justificativa curta explicando o score atribuído.

A Avaliação deve ser um objeto JSON, com as chaves score e raciocinio.

Exemplos:

1.
### Questão:
Os rendimentos, decorrentes da prestação de serviços de consultoria técnica, pagos por pessoa jurídica domiciliada no Brasil a sua matriz no exterior estão sujeitos à legislação de preços de transferência?

### Resposta do Especialista:
Em primeiro lugar, há que distinguir se a prestação dos serviços no Brasil implicou transferência de tecnologia. Na hipótese de ficar comprovada a transferência de tecnologia, com a anuência do Instituto Nacional de Propriedade Industrial (INPI), a transação não estará sujeita às regras de preços de transferência consoante o estabelecido pelo art. 55 da IN RFB nº 1.312, de 2012. Nessa hipótese, a dedução de tais despesas está sujeita aos limites estabelecidos pelos arts. 362 a 365 do RIR/2018. Caso inexista transferência de tecnologia, esses serviços passam a se submeter às regras de preços de transferência.

### Resposta da IA:
Sim, estão sujeitos, a menos que haja transferência de tecnologia com anuência do INPI.

### Avaliação:
{{
  "score": 5,
  "raciocinio": "A resposta da IA responde corretamente à questão principal e menciona a exceção da transferência de tecnologia com anuência do INPI, sem contradição com a resposta especialista."
}}

2.
### Questão:
O que deve ser considerado como "contraprestações creditadas"?

### Resposta do Especialista:
Para efeito do disposto no art. 175 da Instrução Normativa RFB nº 1.700, de 2017, consideram-se contraprestações creditadas as contraprestações vencidas.

### Resposta da IA:
As contraprestações vencidas.

### Avaliação:
{{
  "score": 5,
  "raciocinio": "A resposta da IA apresenta exatamente a informação necessária para responder à questão, sem contradição ou omissão relevante."
}}

3.
### Questão:
Existe prazo para a compensação de prejuízos fiscais da atividade rural?

### Resposta do Especialista:
Não existe qualquer prazo para compensação de prejuízos fiscais da atividade rural.

### Resposta da IA:
O prazo é de 7 dias a partir da data do prejuízo, podendo se extender até 30 dias para compensação de prejuízos fiscais da atividade rural.

### Avaliação:
{{
  "score": 0,
  "raciocinio": "A resposta da IA contradiz diretamente a resposta especialista, pois afirma que existe prazo quando a resposta correta informa que não há prazo."
}}

4.
### Questão:
A base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes?

### Resposta do Especialista:
Sim. A base de cálculo da CSLL, quando negativa, poderá ser compensada até o limite de 30% dos resultados apurados em períodos subsequentes, ajustados pelas adições e exclusões previstas na legislação.

### Resposta da IA:
Não. A base de cálculo da CSLL poderá ser compensada dos resultados apurados em períodos subsequentes.

### Avaliação:
{{
  "score": 0,
  "raciocinio": "A resposta da IA é contraditória e inicia afirmando 'Não', contrariando a resposta especialista, que afirma a possibilidade de compensação da base negativa da CSLL."
}}

5.
### Questão:
A base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes?

### Resposta do Especialista:
Sim. A base de cálculo da CSLL, quando negativa, poderá ser compensada até o limite de 30% dos resultados apurados em períodos subsequentes, ajustados pelas adições e exclusões previstas na legislação.

### Resposta da IA:
Sim, a base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes.

### Avaliação:
{{
  "score": 3,
  "raciocinio": "A resposta da IA acerta a possibilidade de compensação, mas omite informações juridicamente relevantes, como o limite de 30% e os ajustes por adições e exclusões."
}}

6.
### Questão:
A base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes?

### Resposta do Especialista:
Sim. A base de cálculo da CSLL, quando negativa, poderá ser compensada até o limite de 30% dos resultados apurados em períodos subsequentes, ajustados pelas adições e exclusões previstas na legislação.

### Resposta da IA:
Sim, a base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes, até o limite de 30%. Além disso, essa compensação deve observar os ajustes previstos na legislação aplicável, como adições e exclusões.

### Avaliação:
{{
  "score": 5,
  "raciocinio": "A resposta da IA cobre os elementos essenciais da resposta especialista e as informações adicionais estão alinhadas ao conteúdo jurídico esperado."
}}

7.
### Questão:
A base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes?

### Resposta do Especialista:
Sim. A base de cálculo da CSLL, quando negativa, poderá ser compensada até o limite de 30% dos resultados apurados em períodos subsequentes, ajustados pelas adições e exclusões previstas na legislação.

### Resposta da IA:
Sim, a base negativa da CSLL pode ser compensada em períodos subsequentes até o limite de 30%. Essa compensação também pode envolver outros procedimentos fiscais, contábeis e administrativos, dependendo do regime tributário, da escrituração, da atividade da empresa e de eventuais obrigações acessórias aplicáveis.

### Avaliação:
{{
  "score": 4,
  "raciocinio": "A resposta da IA cobre o ponto principal e não contradiz a resposta especialista, mas traz informações adicionais genéricas e pouco necessárias."
}}

8.
### Questão:
A base de cálculo negativa da CSLL poderá ser compensada com resultados apurados em períodos subsequentes?

### Resposta do Especialista:
Sim. A base de cálculo da CSLL, quando negativa, poderá ser compensada até o limite de 30% dos resultados apurados em períodos subsequentes, ajustados pelas adições e exclusões previstas na legislação.

### Resposta da IA:
Sim, a base negativa da CSLL pode ser compensada em períodos subsequentes, mas isso depende de autorização prévia da Receita Federal e pode ser feito integralmente quando a empresa comprovar prejuízo fiscal acumulado.

### Avaliação:
{{
  "score": 2,
  "raciocinio": "A resposta acerta a possibilidade de compensação, mas adiciona informações problemáticas que não aparecem na resposta especialista, como autorização prévia e compensação integral."
}}

Agora faça a Avaliação:

### Questão:
{questao}

### Resposta do Especialista:
{resposta_especialista}

### Resposta da IA:
{resposta_ia}

### Avaliação:
"""


# =========================
# CLIENTS
# =========================

def make_openai_client() -> OpenAI:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "COLOQUE_SUA_CHAVE_OPENAI_AQUI":
        raise RuntimeError("Informe a chave da OpenAI em OPENAI_API_KEY.")
    return OpenAI(api_key=OPENAI_API_KEY)


def make_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "COLOQUE_SUA_CHAVE_GEMINI_AQUI":
        raise RuntimeError("Informe a chave da Gemini em GEMINI_API_KEY.")
    return genai.Client(api_key=GEMINI_API_KEY)


def make_anthropic_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "COLOQUE_SUA_CHAVE_CLAUDE_AQUI":
        raise RuntimeError("Informe a chave da Claude em ANTHROPIC_API_KEY.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


CLIENTS = {
    "openai": make_openai_client(),
    "gemini": make_gemini_client(),
    "anthropic": make_anthropic_client(),
}


# =========================
# JSON / PARSE
# =========================

def clean_json_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_any(text: str) -> Any:
    if not text:
        raise ValueError("Resposta vazia do julgador.")

    text = clean_json_text(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start_obj = text.find("{")
    end_obj = text.rfind("}")

    start_arr = text.find("[")
    end_arr = text.rfind("]")

    candidates = []

    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidates.append(text[start_obj:end_obj + 1])

    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        candidates.append(text[start_arr:end_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError(f"Não foi possível extrair JSON da resposta: {text[:1000]!r}")


def parse_score(value: Any) -> float:
    if value is None:
        return np.nan

    if isinstance(value, str):
        value = value.strip().replace(",", ".")

        # Caso venha algo como "4/5"
        if "/" in value:
            value = value.split("/")[0].strip()

    try:
        score = float(value)
    except Exception:
        return np.nan

    if score < SCORE_MIN:
        score = SCORE_MIN

    if score > SCORE_MAX:
        score = SCORE_MAX

    return score


def sanitize_judge_json(data: dict[str, Any]) -> dict[str, Any]:
    score = parse_score(data.get("score"))

    raciocinio = (
        data.get("raciocinio")
        or data.get("raciocínio")
        or data.get("justificativa")
        or ""
    )

    return {
        "score": score,
        "score_normalizado": score / SCORE_MAX if pd.notna(score) else np.nan,
        "raciocinio": str(raciocinio).strip(),
    }


def score_para_categoria(score: float) -> str:
    if pd.isna(score):
        return "<<ERRO>>"

    if score >= 4:
        return "CORRETO"

    if score >= 3:
        return "PARCIALMENTE_CORRETO"

    return "INCORRETO"


# =========================
# PROMPT EM LOTE
# =========================

def get_prompt_base_for_batch() -> str:
    base = prompt_eval_score.split("Agora faça a Avaliação:")[0]
    base = base.replace("{{", "{").replace("}}", "}")
    return base.strip()


def build_batch_prompt(rows: list[tuple[int, pd.Series]]) -> str:
    itens = []

    for row_idx, row in rows:
        itens.append(
            {
                "id": int(row_idx),
                "questao": str(row["Q"]),
                "resposta_especialista": str(row["A"]),
                "resposta_ia": str(row["A_model"]),
            }
        )

    prompt_base = get_prompt_base_for_batch()

    return f"""
{prompt_base}

Agora avalie TODOS os itens abaixo.

Para cada item, compare:
- questao
- resposta_especialista
- resposta_ia

Retorne APENAS um JSON válido neste formato:

{{
  "avaliacoes": [
    {{
      "id": 0,
      "score": 5,
      "raciocinio": "..."
    }},
    {{
      "id": 1,
      "score": 2,
      "raciocinio": "..."
    }}
  ]
}}

Regras obrigatórias:
- O campo "id" deve ser exatamente o mesmo id recebido.
- O campo "score" deve ser um número de 0 a 5.
- Use apenas a escala definida nas instruções.
- Não retorne CORRETO ou ERRADO como resultado principal.
- Não retorne texto fora do JSON.
- Não pule nenhum item.
- Retorne uma avaliação para cada item recebido.
- O campo "raciocinio" deve justificar brevemente o score.

Itens para avaliar:

{json.dumps(itens, ensure_ascii=False, indent=2)}
""".strip()


# =========================
# CHAMADAS DAS APIS
# =========================

def call_openai_batch(model: str, prompt: str) -> str:
    client: OpenAI = CLIENTS["openai"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            max_completion_tokens=16000,
            messages=messages,
        )
    except Exception as e:
        error_text = str(e).lower()

        if "max_completion_tokens" in error_text or "unsupported" in error_text:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
        else:
            raise

    return response.choices[0].message.content or ""


def call_gemini_batch(model: str, prompt: str) -> str:
    client: genai.Client = CLIENTS["gemini"]

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=16000,
        ),
    )

    return response.text or ""


def call_anthropic_batch(model: str, prompt: str) -> str:
    client: anthropic.Anthropic = CLIENTS["anthropic"]

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    texts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)

    return "\n".join(texts)


def call_judge_batch(provider: str, model: str, prompt: str) -> str:
    if provider == "openai":
        return call_openai_batch(model, prompt)

    if provider == "gemini":
        return call_gemini_batch(model, prompt)

    if provider == "anthropic":
        return call_anthropic_batch(model, prompt)

    raise ValueError(f"Provider desconhecido: {provider}")


# =========================
# PARSE DO LOTE
# =========================

def parse_batch_response(
    raw: str,
    expected_ids: set[int],
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    data = extract_json_any(raw)

    if isinstance(data, list):
        avaliacoes = data
    elif isinstance(data, dict):
        avaliacoes = data.get("avaliacoes", [])
    else:
        raise ValueError("JSON do julgador não é objeto nem lista.")

    if not isinstance(avaliacoes, list):
        raise ValueError("Campo 'avaliacoes' não é uma lista.")

    results_by_id = {}
    raw_by_id = {}

    for item in avaliacoes:
        if not isinstance(item, dict):
            continue

        if "id" not in item:
            continue

        row_id = int(item["id"])

        result = sanitize_judge_json(item)

        if pd.isna(result["score"]):
            raise ValueError(f"Score inválido no lote: {item}")

        results_by_id[row_id] = result
        raw_by_id[row_id] = json.dumps(item, ensure_ascii=False)

    returned_ids = set(results_by_id.keys())
    missing = expected_ids - returned_ids

    if missing:
        raise ValueError(f"O julgador não retornou avaliação para os ids: {sorted(missing)}")

    return results_by_id, raw_by_id


# =========================
# RETRY / ERROS
# =========================

def is_non_retryable_api_error(error_text: str) -> bool:
    error_text = error_text.lower()

    patterns = [
        "credit balance is too low",
        "resource_exhausted",
        "quota exceeded",
        "exceeded your current quota",
        "limit: 0",
        "free_tier_requests",
        "free_tier_input_token_count",
        "generate_requests_per_model_per_day",
        "billing",
        "insufficient_quota",
    ]

    return any(pattern in error_text for pattern in patterns)


def get_retry_sleep_seconds(error_text: str, attempt: int) -> float:
    error_text = error_text.lower()

    if "502" in error_text or "bad gateway" in error_text:
        return 30 + random.uniform(0, 5)

    if "503" in error_text or "service unavailable" in error_text:
        return 45 + random.uniform(0, 10)

    if "504" in error_text or "deadline" in error_text or "timeout" in error_text:
        return 45 + random.uniform(0, 10)

    return min(60, 2 ** (attempt - 1)) + random.uniform(0, 3)


def make_error_batch_result(
    rows: list[tuple[int, pd.Series]],
    error: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[int, str]]:
    results_by_id = {}
    raw_by_id = {}
    error_by_id = {}

    for row_idx, _ in rows:
        results_by_id[row_idx] = {
            "score": np.nan,
            "score_normalizado": np.nan,
            "raciocinio": "",
        }
        raw_by_id[row_idx] = ""
        error_by_id[row_idx] = error

    return results_by_id, raw_by_id, error_by_id


def judge_batch_with_retry(
    judge_name: str,
    rows: list[tuple[int, pd.Series]],
) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[int, str]]:
    if judge_name in DISABLED_JUDGES:
        return make_error_batch_result(
            rows,
            f"Julgador {judge_name} desativado por erro anterior de quota/crédito.",
        )

    config = JUDGES[judge_name]
    provider = config["provider"]
    model = config["model"]

    prompt = build_batch_prompt(rows)
    expected_ids = {row_idx for row_idx, _ in rows}

    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_judge_batch(provider, model, prompt)
            results_by_id, raw_by_id = parse_batch_response(raw, expected_ids)

            error_by_id = {
                row_idx: ""
                for row_idx, _ in rows
            }

            return results_by_id, raw_by_id, error_by_id

        except Exception as e:
            last_error = str(e)
            print(f"[{judge_name}-score-batch] tentativa {attempt}/{MAX_RETRIES} falhou: {last_error}")

            if is_non_retryable_api_error(last_error):
                print(f"[{judge_name}-score-batch] erro de quota/crédito. Desativando esse julgador.")
                DISABLED_JUDGES.add(judge_name)
                break

            sleep_seconds = get_retry_sleep_seconds(last_error, attempt)
            print(f"[{judge_name}-score-batch] aguardando {sleep_seconds:.1f}s antes de tentar novamente...")
            time.sleep(sleep_seconds)

    return make_error_batch_result(rows, last_error)


# =========================
# RESULTADO OFICIAL
# =========================

def add_judge_result(
    df: pd.DataFrame,
    row_idx: int,
    judge_name: str,
    result: dict[str, Any],
    raw: str,
    error: str,
) -> None:
    prefix = f"{judge_name}_judge"

    df.at[row_idx, f"{prefix}_score_0_5"] = result["score"]
    df.at[row_idx, f"{prefix}_score_normalizado"] = result["score_normalizado"]
    df.at[row_idx, f"{prefix}_categoria"] = score_para_categoria(result["score"])
    df.at[row_idx, f"{prefix}_raciocinio"] = result["raciocinio"]
    df.at[row_idx, f"{prefix}_raw"] = raw
    df.at[row_idx, f"{prefix}_erro"] = error


def compute_official_result_for_row(df: pd.DataFrame, row_idx: int) -> None:
    scores = []
    categorias = {}

    for judge_name in JUDGES:
        score_col = f"{judge_name}_judge_score_0_5"
        cat_col = f"{judge_name}_judge_categoria"

        if score_col in df.columns:
            score = pd.to_numeric(pd.Series([df.at[row_idx, score_col]]), errors="coerce").iloc[0]

            if pd.notna(score):
                scores.append(float(score))

        if cat_col in df.columns:
            categorias[judge_name] = str(df.at[row_idx, cat_col])
        else:
            categorias[judge_name] = ""

    if not scores:
        official_score = np.nan
        official_score_norm = np.nan
        official_category = "<<ERRO>>"
        decision_rule = "sem_scores_validos"
    else:
        official_score = float(np.mean(scores))
        official_score_norm = official_score / SCORE_MAX
        official_category = score_para_categoria(official_score)

        if len(scores) == 1:
            decision_rule = "apenas_um_score_valido"
        else:
            decision_rule = "media_dos_scores_validos"

    df.at[row_idx, "score_oficial_0_5"] = official_score
    df.at[row_idx, "score_oficial_normalizado"] = official_score_norm
    df.at[row_idx, "resultado_oficial"] = official_category
    df.at[row_idx, "regra_decisao"] = decision_rule

    votos_scores = {}

    for judge_name in JUDGES:
        col = f"{judge_name}_judge_score_0_5"
        if col in df.columns:
            votos_scores[judge_name] = df.at[row_idx, col]
        else:
            votos_scores[judge_name] = ""

    df.at[row_idx, "scores_julgadores"] = json.dumps(votos_scores, ensure_ascii=False)
    df.at[row_idx, "categorias_julgadores"] = json.dumps(categorias, ensure_ascii=False)


def compute_official_results_for_all_rows(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        compute_official_result_for_row(df, i)


def count_unanimity_and_disagreement(df: pd.DataFrame) -> tuple[int, int]:
    n_unanimidade = 0
    n_divergencia = 0

    cat_cols = [
        f"{judge_name}_judge_categoria"
        for judge_name in JUDGES
        if f"{judge_name}_judge_categoria" in df.columns
    ]

    for _, row in df.iterrows():
        cats = [
            str(row[col])
            for col in cat_cols
            if pd.notna(row[col]) and str(row[col]) not in ["", "<<ERRO>>"]
        ]

        if len(cats) < 2:
            continue

        if len(set(cats)) == 1:
            n_unanimidade += 1
        else:
            n_divergencia += 1

    return n_unanimidade, n_divergencia


def compute_score_std(df: pd.DataFrame) -> pd.Series:
    score_cols = [
        f"{judge_name}_judge_score_0_5"
        for judge_name in JUDGES
        if f"{judge_name}_judge_score_0_5" in df.columns
    ]

    if not score_cols:
        return pd.Series([np.nan] * len(df))

    return df[score_cols].apply(pd.to_numeric, errors="coerce").std(axis=1)


# =========================
# EXECUÇÃO POR MODELO
# =========================

def run_all_judges_in_batches(df: pd.DataFrame, out_path: Path) -> None:
    all_rows = [(i, df.iloc[i]) for i in range(len(df))]

    for judge_name in JUDGES:
        print(f"\nRodando {judge_name} em lotes de {JUDGE_BATCH_SIZE} perguntas...")

        for start in tqdm(
            range(0, len(all_rows), JUDGE_BATCH_SIZE),
            desc=f"{judge_name} score batch",
            unit="batch",
        ):
            batch_rows = all_rows[start:start + JUDGE_BATCH_SIZE]

            results_by_id, raw_by_id, error_by_id = judge_batch_with_retry(
                judge_name=judge_name,
                rows=batch_rows,
            )

            for row_idx, _ in batch_rows:
                result = results_by_id.get(
                    row_idx,
                    {
                        "score": np.nan,
                        "score_normalizado": np.nan,
                        "raciocinio": "",
                    },
                )

                add_judge_result(
                    df=df,
                    row_idx=row_idx,
                    judge_name=judge_name,
                    result=result,
                    raw=raw_by_id.get(row_idx, ""),
                    error=error_by_id.get(row_idx, ""),
                )

            df.to_csv(out_path, index=False, encoding="utf-8")

            if SLEEP_BETWEEN_BATCHES > 0:
                time.sleep(SLEEP_BETWEEN_BATCHES)


# =========================
# ARQUIVOS
# =========================

def extract_model_name(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_QAG", "")
    stem = stem.replace("_judged", "")
    stem = stem.replace("_gpt4-judge", "")
    stem = stem.replace("_NEW", "")
    return stem


def judge_file(path: Path) -> Path:
    model_name = extract_model_name(path)
    out_path = OUTPUT_DIR / f"{path.stem}{OUTPUT_SUFFIX}"

    print(f"\n=== Julgando arquivo: {path.name} ===")
    print(f"Modelo avaliado: {model_name}")
    print(f"Rodando as {N_QUESTOES_TESTE} primeiras perguntas.")
    print(f"Batch size: {JUDGE_BATCH_SIZE}")
    print(f"Saída: {out_path}")

    df_original = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df_original.columns]

    if missing_cols:
        print(f">> Pulando {path.name}. Faltam colunas: {missing_cols}")
        return out_path

    df = df_original.head(N_QUESTOES_TESTE).copy()
    df.reset_index(drop=True, inplace=True)

    df["modelo_avaliado"] = model_name
    df["prompt_version"] = PROMPT_VERSION

    run_all_judges_in_batches(df, out_path)

    compute_official_results_for_all_rows(df)

    df["desvio_score_entre_julgadores"] = compute_score_std(df)

    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f">> Arquivo salvo em: {out_path.resolve()}")

    return out_path


# =========================
# SUMMARY
# =========================

def build_summary(judged_files: list[Path]) -> Path:
    rows = []

    for path in judged_files:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        model_name = str(df["modelo_avaliado"].iloc[0])

        score_oficial = pd.to_numeric(df["score_oficial_0_5"], errors="coerce")
        score_norm = pd.to_numeric(df["score_oficial_normalizado"], errors="coerce")

        n_unanimidade, n_divergencia = count_unanimity_and_disagreement(df)

        row = {
            "modelo": model_name,
            "arquivo": path.name,
            "n_total": len(df),
            "score_medio_0_5": score_oficial.mean(),
            "score_medio_normalizado": score_norm.mean(),
            "n_correto": int((df["resultado_oficial"] == "CORRETO").sum()),
            "n_parcialmente_correto": int((df["resultado_oficial"] == "PARCIALMENTE_CORRETO").sum()),
            "n_incorreto": int((df["resultado_oficial"] == "INCORRETO").sum()),
            "n_erros": int((df["resultado_oficial"] == "<<ERRO>>").sum()),
            "n_unanimidade_categoria": n_unanimidade,
            "n_divergencia_categoria": n_divergencia,
            "taxa_unanimidade_categoria": n_unanimidade / len(df),
            "taxa_divergencia_categoria": n_divergencia / len(df),
            "desvio_medio_score_entre_julgadores": pd.to_numeric(
                df.get("desvio_score_entre_julgadores"),
                errors="coerce",
            ).mean(),
        }

        for judge_name in JUDGES:
            row[f"media_{judge_name}_score_0_5"] = pd.to_numeric(
                df.get(f"{judge_name}_judge_score_0_5"),
                errors="coerce",
            ).mean()

            row[f"media_{judge_name}_score_normalizado"] = pd.to_numeric(
                df.get(f"{judge_name}_judge_score_normalizado"),
                errors="coerce",
            ).mean()

            row[f"n_{judge_name}_erros"] = int(
                (df.get(f"{judge_name}_judge_erro", pd.Series([""] * len(df)))
                 .fillna("")
                 .astype(str) != "").sum()
            )

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            "score_medio_0_5",
            ascending=False,
        )
        summary_df.reset_index(drop=True, inplace=True)

    summary_path = OUTPUT_DIR / "summary_101_3_julgadores_score_batch.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n=== RESUMO 101 PERGUNTAS — 3 JULGADORES SCORE 0 A 5 EM LOTE ===")
    print(summary_df)
    print(f"\nResumo salvo em: {summary_path.resolve()}")

    return summary_path


def build_all_results_file(judged_files: list[Path]) -> Path:
    all_dfs = []

    for path in judged_files:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        df["arquivo_origem"] = path.name
        all_dfs.append(df)

    out_path = OUTPUT_DIR / "all_results_101_3_julgadores_score_batch.csv"

    if not all_dfs:
        print("Nenhum resultado detalhado para juntar.")
        return out_path

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nArquivo único com todos os resultados salvo em: {out_path.resolve()}")

    return out_path


def build_files_by_judge(judged_files: list[Path]) -> list[Path]:
    output_paths = []

    base_cols = [
        "modelo_avaliado",
        "prompt_version",
        "Q",
        "A",
        "A_model",
        "score_oficial_0_5",
        "score_oficial_normalizado",
        "resultado_oficial",
        "regra_decisao",
        "scores_julgadores",
        "categorias_julgadores",
        "desvio_score_entre_julgadores",
    ]

    for judge_name in JUDGES:
        all_dfs = []

        judge_cols = [
            f"{judge_name}_judge_score_0_5",
            f"{judge_name}_judge_score_normalizado",
            f"{judge_name}_judge_categoria",
            f"{judge_name}_judge_raciocinio",
            f"{judge_name}_judge_raw",
            f"{judge_name}_judge_erro",
        ]

        for path in judged_files:
            if not path.exists():
                continue

            df = pd.read_csv(path)

            if df.empty:
                continue

            cols_to_keep = [
                col for col in base_cols + judge_cols
                if col in df.columns
            ]

            df_judge = df[cols_to_keep].copy()
            df_judge["modelo_julgador"] = judge_name
            df_judge["arquivo_origem"] = path.name

            all_dfs.append(df_judge)

        judge_output_path = OUTPUT_DIR / f"resultados_{judge_name}_score_101_batch.csv"

        if not all_dfs:
            print(f">> Nenhum resultado encontrado para o julgador {judge_name}")
            continue

        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(judge_output_path, index=False, encoding="utf-8")

        print(f">> Arquivo do julgador {judge_name} salvo em: {judge_output_path.resolve()}")

        output_paths.append(judge_output_path)

    return output_paths


# =========================
# MAIN
# =========================

def main() -> None:
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"Diretório não existe: {CSV_DIR.resolve()}")

    files = sorted(CSV_DIR.glob("*_QAG.csv"))

    if not files:
        print(f"Nenhum arquivo *_QAG.csv encontrado em {CSV_DIR.resolve()}")
        return

    print(f"Arquivos encontrados em {CSV_DIR.resolve()}:")
    for file in files:
        print(" -", file.name)

    judged_files = []

    for file in files:
        out_path = judge_file(file)
        judged_files.append(out_path)

    build_summary(judged_files)
    build_all_results_file(judged_files)
    build_files_by_judge(judged_files)


if __name__ == "__main__":
    main()