"""
mult_judge_binario_101_batch.py

Avaliação automática de respostas de LLM usando GPT, Gemini e Claude como julgadores.

- Usa o prompt antigo
- Julgamento binário: CORRETO / ERRADO
- Para cada modelo avaliado, julga as 101 perguntas em lote
- Depois junta os resultados dos 3 julgadores no mesmo CSV
- Calcula resultado oficial por maioria

Entrada:
    results/legacy/QAG/aberto/*_QAG.csv

Saída:
    results/legacy/julgamento_binario_101_3_julgadores_batch/

Colunas esperadas no CSV:
    Q, A, A_model

Instalar:
    pip install openai google-genai anthropic pandas tqdm numpy
"""

from pathlib import Path
from collections import Counter
import os
import json
import re
import time
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Coloque suas chaves aqui.
# CUIDADO: não suba esse arquivo para GitHub com as chaves preenchidas.



# =========================
# CONFIGURAÇÕES
# =========================

CSV_DIR = Path("results/legacy/QAG/aberto")
OUTPUT_DIR = Path("results/legacy/result_judges")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_QUESTOES_TESTE = 101

# 101 = tenta julgar todas as perguntas do arquivo em uma única chamada por julgador.
# Se algum julgador cortar o JSON ou falhar por tamanho, diminua para 50, 25 ou 20.
JUDGE_BATCH_SIZE = 25

REQUIRED_COLS = ["Q", "A", "A_model"]

OUTPUT_SUFFIX = "result_julgadores.csv"
PROMPT_VERSION = "prompt_antigo_binario_3_julgadores_101_batch"

SLEEP_BETWEEN_BATCHES = 0.0
MAX_RETRIES = 3

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

# FIX: padronizado para CORRETO / ERRADO em todo o código
VALID_RESULTS = {
    "CORRETO",
    "ERRADO",
}

SCORE_BY_RESULT = {
    "CORRETO": 1.0,
    "ERRADO": 0.0,
}

SYSTEM_PROMPT = (
    "Você é um avaliador jurídico especialista em direito tributário brasileiro. "
    "Siga rigorosamente as instruções do usuário e retorne apenas um JSON válido."
)

# Se um julgador der erro de quota/crédito, ele entra aqui e é pulado nas próximas chamadas.
DISABLED_JUDGES = set()


# =========================
# PROMPT ANTIGO
# =========================

prompt_eval = """
Instruções:
Avalie a resposta gerada pela IA com base nos seguintes critérios:

1. Verifique se a Resposta da IA está contida na resposta Resposta do Especialista, ou seja, se não existe nenhuma contradição. Ignore termos diferentes ou pequenas informações a mais ou a menos.
2. A Resposta do Especialista pode conter mais informações do que foi solicitado na pergunta, se a informação da Resposta do Especialista não for necessária para responder a pergunta não use para avaliar a Resposta da IA.
3. Se a Resposta da IA tiver mais informações do que a Resposta do Especialista não deve ser levada em consideração para avaliação desde que as informações estejam corretas.
3. Verifique se a resposta pode responder à questão. Para isso veja se a resposta fornece as informações solicitadas na questão, se é suficiente. Por exemplo, se a questão pode ser respondida com um simples "Não", isso é aceitável.

Inclua um raciocínio que justifique a Avaliação. Se os critérios forem atendidos, retorne 'CORRETO'. Se qualquer um dos critérios não for atendido, retorne 'ERRADO'.

A Avaliação deve ser um objeto JSON, com as chaves resultado e raciocínio.

Exemplos:

1. 
### Questão:
Os rendimentos, decorrentes da prestação de serviços de consultoria técnica,  pagos por pessoa jurídica domiciliada no Brasil a sua matriz no exterior estão  sujeitos à legislação de preços de transferência?

### Resposta do Especialista:
Em primeiro lugar, há que distinguir se a prestação dos serviços no Brasil implicou transferência de tecnologia. Na hipótese de ficar comprovada a transferência de tecnologia, com a anuência do Instituto Nacional de Propriedade Industrial (INPI), a transação não estará sujeita às regras de preços de transferência consoante o estabelecido pelo art. 55 da IN RFB nº 1.312, de 2012. Nessa hipótese, a dedução de tais despesas está sujeita aos limites estabelecidos pelos arts. 362 a 365 do RIR/2018. Caso inexista transferência de tecnologia, esses serviços passam a se submeter às regras de preços de transferência.

### Resposta da IA:
Sim, estão sujeitos, a menos que haja transferência de tecnologia com anuência do INPI.

### Avaliação:
{{
  "raciocinio": "A resposta da IA está alinhada com a resposta do especialista, abordando corretamente a questão sem contradições, embora seja mais curta.",
  "resultado": "CORRETO"
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
  "raciocinio": "A resposta da IA cobre as principais hipóteses mencionadas pelo especialista, sem apresentar contradições, embora seja menos detalhada.",
  "resultado": "CORRETO"
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
  "raciocinio": "A resposta da IA está incorreta porque menciona '7 dias' e 'até 30 dias', que contradizem a resposta do especialista.",
  "resultado": "ERRADO"
}}

4. 
### Questão:
A base de cálculo negativa da CSLL poderá ser compensada com resultados  apurados em períodos subsequentes?

### Resposta do Especialista:
Sim. A base de cálculo da CSLL, quando negativa, poderá ser compensada até o limite de 30% dos resultados apurados em períodos subsequentes, ajustados pelas adições e exclusões previstas na legislação.

### Resposta da IA:
Não. A base de cálculo da CSLL poderá ser compensada dos resultados apurados em períodos subsequentes.

### Avaliação:
{{
  "raciocinio": "A resposta da IA contradiz a resposta do especialista, fornecendo uma informação oposta sobre a necessidade possibilidade de compensação da CSLL.",
  "resultado": "ERRADO"
}}

Agora pense passo a passo e faça essa Avaliação:

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


# FIX: inicialização lazy — só cria o client quando for usar, evitando falha na importação
def get_client(provider: str):
    if provider == "openai":
        return make_openai_client()
    if provider == "gemini":
        return make_gemini_client()
    if provider == "anthropic":
        return make_anthropic_client()
    raise ValueError(f"Provider desconhecido: {provider}")


# =========================
# UTILITÁRIOS DE JSON
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


def normalize_result(value: Any) -> str:
    raw = str(value or "").strip().upper()
    raw = raw.replace(" ", "_").replace("-", "_")

    if raw.startswith("CORRETO"):
        return "CORRETO"

    # FIX: normaliza tanto ERRADO quanto INCORRETO → ERRADO
    if raw.startswith("ERRADO") or raw.startswith("INCORRETO"):
        return "ERRADO"

    return "<<ERRO>>"


def sanitize_judge_json(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "resultado": normalize_result(data.get("resultado")),
        "raciocinio": str(data.get("raciocinio", "")).strip(),
    }


def score_result(resultado: str) -> float:
    # FIX: usa SCORE_BY_RESULT como fonte única de verdade (inclui ERRADO)
    return SCORE_BY_RESULT.get(resultado, np.nan)


# =========================
# PROMPT EM LOTE
# =========================

def get_prompt_base_for_batch() -> str:
    """
    Usa a mesma parte de instruções e exemplos do prompt antigo.
    Remove a parte final que era para uma única pergunta.
    """

    base = prompt_eval.split("Agora pense passo a passo")[0]

    # No prompt original, as chaves estão duplicadas por causa do .format().
    # Para mandar direto ao modelo, voltamos para chaves normais.
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
      "resultado": "CORRETO",
      "raciocinio": "..."
    }},
    {{
      "id": 1,
      "resultado": "ERRADO",
      "raciocinio": "..."
    }}
  ]
}}

Regras obrigatórias:
- O campo "id" deve ser exatamente o mesmo id recebido.
- O campo "resultado" deve ser apenas "CORRETO" ou "ERRADO".
- Não use "PARCIALMENTE_CORRETO" nem "INCORRETO".
- Não retorne texto fora do JSON.
- Não pule nenhum item.
- Retorne uma avaliação para cada item recebido.
- O campo "raciocinio" deve ser curto, mas suficiente para justificar o julgamento.

Itens para avaliar:

{json.dumps(itens, ensure_ascii=False, indent=2)}
""".strip()


# =========================
# CHAMADAS EM LOTE
# =========================

def call_openai_batch(model: str, prompt: str) -> str:
    client: OpenAI = get_client("openai")

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

        # Alguns modelos/SDKs podem não aceitar max_completion_tokens.
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
    client: genai.Client = get_client("gemini")

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
    client: anthropic.Anthropic = get_client("anthropic")

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
# PARSE DO RESULTADO EM LOTE
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

        if result["resultado"] not in VALID_RESULTS:
            raise ValueError(f"Resultado inválido no lote: {result}")

        results_by_id[row_id] = result
        raw_by_id[row_id] = json.dumps(item, ensure_ascii=False)

    returned_ids = set(results_by_id.keys())
    missing = expected_ids - returned_ids

    if missing:
        raise ValueError(f"O julgador não retornou avaliação para os ids: {sorted(missing)}")

    return results_by_id, raw_by_id


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
    ]

    return any(pattern in error_text for pattern in patterns)


def make_error_batch_result(
    rows: list[tuple[int, pd.Series]],
    error: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, str], dict[int, str]]:
    results_by_id = {}
    raw_by_id = {}
    error_by_id = {}

    for row_idx, _ in rows:
        results_by_id[row_idx] = {
            "resultado": "<<ERRO>>",
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
            print(f"[{judge_name}-batch] tentativa {attempt}/{MAX_RETRIES} falhou: {last_error}")

            if is_non_retryable_api_error(last_error):
                print(f"[{judge_name}-batch] erro de quota/crédito. Desativando esse julgador.")
                DISABLED_JUDGES.add(judge_name)
                break

            time.sleep(2 ** (attempt - 1))

    return make_error_batch_result(rows, last_error)


# =========================
# RESULTADO OFICIAL
# =========================

def decide_official_result(results: list[str]) -> tuple[str, float, str]:
    valid = [r for r in results if r in VALID_RESULTS]

    if not valid:
        return "<<ERRO>>", np.nan, "sem_votos_validos"

    if len(valid) == 1:
        result = valid[0]
        return result, SCORE_BY_RESULT[result], "apenas_um_voto_valido"

    counts = Counter(valid)
    most_common = counts.most_common()

    # Com 3 julgadores, se algum resultado tiver 2 votos, é maioria.
    if most_common[0][1] >= 2:
        result = most_common[0][0]
        return result, SCORE_BY_RESULT[result], "maioria"

    # FIX: usa SCORE_BY_RESULT como fonte única — SEM_CONSENSO retorna NaN consistentemente
    return "SEM_CONSENSO", np.nan, "discordancia_sem_maioria"


def add_judge_result(
    df: pd.DataFrame,
    row_idx: int,
    judge_name: str,
    result: dict[str, Any],
    raw: str,
    error: str,
) -> None:
    prefix = f"{judge_name}_judge"

    df.at[row_idx, f"{prefix}_resultado"] = result["resultado"]
    df.at[row_idx, f"{prefix}_score"] = score_result(result["resultado"])
    df.at[row_idx, f"{prefix}_raciocinio"] = result["raciocinio"]
    df.at[row_idx, f"{prefix}_raw"] = raw
    df.at[row_idx, f"{prefix}_erro"] = error


def compute_official_result_for_row(df: pd.DataFrame, row_idx: int) -> None:
    results = []

    for judge_name in JUDGES:
        col = f"{judge_name}_judge_resultado"

        if col in df.columns and pd.notna(df.at[row_idx, col]):
            results.append(str(df.at[row_idx, col]).strip())

    official_result, official_score, decision_rule = decide_official_result(results)

    df.at[row_idx, "resultado_oficial"] = official_result
    df.at[row_idx, "score_oficial"] = official_score
    df.at[row_idx, "regra_decisao"] = decision_rule

    votes = {}

    for judge_name in JUDGES:
        col = f"{judge_name}_judge_resultado"
        votes[judge_name] = str(df.at[row_idx, col]) if col in df.columns else ""

    df.at[row_idx, "votos_julgadores"] = json.dumps(votes, ensure_ascii=False)


def compute_official_results_for_all_rows(df: pd.DataFrame) -> None:
    for i in range(len(df)):
        compute_official_result_for_row(df, i)


def count_unanimity_and_disagreement(df: pd.DataFrame) -> tuple[int, int]:
    n_unanimidade = 0
    n_divergencia = 0

    judge_cols = [
        f"{judge_name}_judge_resultado"
        for judge_name in JUDGES
        if f"{judge_name}_judge_resultado" in df.columns
    ]

    for _, row in df.iterrows():
        votes = [
            str(row[col])
            for col in judge_cols
            if pd.notna(row[col]) and str(row[col]) in VALID_RESULTS
        ]

        if len(votes) < 2:
            continue

        if len(set(votes)) == 1:
            n_unanimidade += 1
        else:
            n_divergencia += 1

    return n_unanimidade, n_divergencia


# =========================
# EXECUÇÃO EM LOTE POR MODELO
# =========================

def run_all_judges_in_batches(df: pd.DataFrame, out_path: Path) -> None:
    all_rows = [(i, df.iloc[i]) for i in range(len(df))]

    for judge_name in JUDGES:
        print(f"\nRodando {judge_name} em lotes de {JUDGE_BATCH_SIZE} perguntas...")

        for start in tqdm(
            range(0, len(all_rows), JUDGE_BATCH_SIZE),
            desc=f"{judge_name} batch",
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
                        "resultado": "<<ERRO>>",
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

            # checkpoint depois de cada lote
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

    df_original = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df_original.columns]

    if missing_cols:
        print(f">> Pulando {path.name}. Faltam colunas: {missing_cols}")
        return out_path

    df = df_original.head(N_QUESTOES_TESTE).copy()
    df.reset_index(drop=True, inplace=True)

    df["modelo_avaliado"] = model_name
    df["prompt_version"] = PROMPT_VERSION

    # 1. Roda GPT, Gemini e Claude em lote.
    run_all_judges_in_batches(df, out_path)

    # 2. Junta os votos e calcula o resultado oficial.
    compute_official_results_for_all_rows(df)

    # 3. Salva o CSV final daquele modelo.
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f">> Arquivo salvo em: {out_path.resolve()}")

    return out_path


def build_summary(judged_files: list[Path]) -> Path:
    rows = []

    for path in judged_files:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if df.empty:
            continue

        model_name = str(df["modelo_avaliado"].iloc[0])
        scores = pd.to_numeric(df["score_oficial"], errors="coerce")

        n_unanimidade, n_divergencia = count_unanimity_and_disagreement(df)

        # FIX: verifica existência da coluna antes de acessar
        def safe_mean(col_name: str) -> float:
            if col_name not in df.columns:
                return np.nan
            return pd.to_numeric(df[col_name], errors="coerce").mean()

        rows.append(
            {
                "modelo": model_name,
                "arquivo": path.name,
                "n_total": len(df),
                "n_correto": int((df["resultado_oficial"] == "CORRETO").sum()),
                "n_errado": int((df["resultado_oficial"] == "ERRADO").sum()),
                "n_sem_consenso": int((df["resultado_oficial"] == "SEM_CONSENSO").sum()),
                "n_erros": int((df["resultado_oficial"] == "<<ERRO>>").sum()),
                "n_maioria": int((df["regra_decisao"] == "maioria").sum()),
                "n_apenas_um_voto_valido": int(
                    (df["regra_decisao"] == "apenas_um_voto_valido").sum()
                ),
                "n_discordancia_sem_maioria": int(
                    (df["regra_decisao"] == "discordancia_sem_maioria").sum()
                ),
                "n_unanimidade": n_unanimidade,
                "n_divergencia_entre_julgadores": n_divergencia,
                "acuracia_binaria_101": scores.mean(),
                "media_gpt_score": safe_mean("gpt_judge_score"),
                "media_gemini_score": safe_mean("gemini_judge_score"),
                "media_claude_score": safe_mean("claude_judge_score"),
            }
        )

    summary_df = pd.DataFrame(rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            "acuracia_binaria_101",
            ascending=False,
        )
        summary_df.reset_index(drop=True, inplace=True)

    summary_path = OUTPUT_DIR / "summary_101_3_julgadores_binario_batch.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n=== RESUMO 101 PERGUNTAS — 3 JULGADORES BINÁRIO EM LOTE ===")
    print(summary_df)
    print(f"\nResumo salvo em: {summary_path.resolve()}")

    return summary_path


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


if __name__ == "__main__":
    main()