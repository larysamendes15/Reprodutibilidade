#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
qa_nim_multi_models.py

Gera respostas usando modelos via NVIDIA NIM/API Catalog,
com chamada OpenAI-compatible.

Configuração:
- temperature = 0.1
- max_tokens = 4096
- sem top_p explícito
- sem system prompt
- mesmo template de prompt para todos os modelos
- saída em CSV com Q, A, A_model
- checkpoint por modelo
- retry com backoff exponencial
- trata resposta vazia como erro e tenta novamente

Como rodar:

    pip install openai datasets pandas tqdm

    python qa_nim_multi_models.py

Rodar só alguns exemplos:

    python qa_nim_multi_models.py --max-rows 5

Continuar de onde parou:

    python qa_nim_multi_models.py --resume

Rodar apenas um modelo pelo alias:

    python qa_nim_multi_models.py --only deepseek
    python qa_nim_multi_models.py --only gemma
    python qa_nim_multi_models.py --only qwen
"""

from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Configurações gerais
# ============================================================

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Cole sua chave aqui localmente.
# Não suba esse arquivo no Git com a chave preenchida.
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
DATASET_DIR = "data/tax_law_brazil_cosit"
SPLIT_NAME = "tax_law"

OUTPUT_DIR = "results/respostas_geradas"

TEMPERATURE = 0.1
MAX_TOKENS = 4096

REQUEST_TIMEOUT_SEC = 180

RETRIES = 6
BASE_SLEEP = 1.0
SLEEP_BETWEEN_CALLS_SEC = 0.0
SAVE_EVERY = 25


# ============================================================
# Modelos via NVIDIA NIM / OpenAI-compatible
# ============================================================

MODELS_TO_RUN = [
    {
        "model": "deepseek-ai/deepseek-v4-flash",
        "alias": "deepseek-ai_deepseek-v4-flash",
    },
    {
        "model": "google/gemma-4-31b-it",
        "alias": "google_gemma-4-31b-it",
    },
    {
        "model": "meta/llama-3.3-70b-instruct",
        "alias": "meta_llama-3.3-70b-instruct",
    },
    {
        "model": "mistralai/mixtral-8x22b-instruct-v0.1",
        "alias": "mistralai_mixtral-8x22b-instruct-v0.1",
    },
    {
        "model": "openai/gpt-oss-120b",
        "alias": "openai_gpt-oss-120b",
    },
    {
        "model": "qwen/qwen3.5-122b-a10b",
        "alias": "qwen_qwen3.5-122b-a10b",
    },
    {
        "model": "z-ai/glm-5.1",
        "alias": "z-ai_glm-5.1",
    },

    # Modelo antigo (rodada anterior). Confirme o ID exato no catálogo da
    # NVIDIA NIM antes de descomentar.
    #
    # {
    #     "model": "qwen/qwen2-72b-instruct",
    #     "alias": "qwen2-72B-Instruct",
    # },
]


# ============================================================
# Prompt baseado no template do Presa
# ============================================================

RETRIEVER_TEMPLATE = """Use the following pieces of legal information from laws to answer the user's question.
If the answer is not clear in context, try to figure out by interpreting the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Do not quote the "contextual information" provided in the answer, do not say "according to the information" or anything like that, use the information only to answer the question.
Only return the helpful answer below and nothing else.
Answer the question in Portuguese.
Helpful answer:"""


# ============================================================
# Cliente
# ============================================================

def get_client() -> OpenAI:
    api_key = NIM_API_KEY

    if not api_key or api_key.strip() == "" or api_key == "COLE_SUA_CHAVE_AQUI":
        raise RuntimeError(
            "NIM_API_KEY não definida.\n"
            "Coloque sua chave no campo NIM_API_KEY no início do arquivo."
        )

    return OpenAI(
        base_url=NIM_BASE_URL,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SEC,
    )


# ============================================================
# Utilitários
# ============================================================

def build_prompt(context: str, question: str) -> str:
    return RETRIEVER_TEMPLATE.format(
        context=context,
        question=question,
    )


def safe_name(alias: str) -> str:
    return alias.replace("/", "_")


def output_path(alias: str) -> Path:
    return Path(OUTPUT_DIR) / f"{safe_name(alias)}_QAG.csv"


def checkpoint_path(alias: str) -> Path:
    return Path(OUTPUT_DIR) / f"{safe_name(alias)}.checkpoint.json"


def load_checkpoint(alias: str) -> int:
    path = checkpoint_path(alias)

    if not path.exists():
        return 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("next_index", 0))
    except Exception:
        return 0


def save_checkpoint(alias: str, next_index: int) -> None:
    path = checkpoint_path(alias)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            {"next_index": next_index},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_existing_rows(alias: str) -> list[dict]:
    path = output_path(alias)

    if not path.exists():
        return []

    return pd.read_csv(path).to_dict(orient="records")


def save_rows(alias: str, rows: list[dict]) -> None:
    path = output_path(alias)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        rows,
        columns=["Q", "A", "A_model"],
    ).to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def save_index(index_rows: list[dict]) -> None:
    index_file = Path(OUTPUT_DIR) / "index_presa_like_nim.csv"

    pd.DataFrame(index_rows).to_csv(
        index_file,
        index=False,
        encoding="utf-8",
    )


def response_to_dict(obj: Any) -> dict:
    """
    Converte objetos da SDK OpenAI/Pydantic em dict, quando possível.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    if isinstance(obj, dict):
        return obj

    return {}


def print_raw_response(resp: Any) -> None:
    """
    Imprime o retorno bruto da API para diagnosticar respostas vazias.
    """
    print("\nResposta bruta da API:", flush=True)

    try:
        if hasattr(resp, "model_dump_json"):
            print(resp.model_dump_json(indent=2), flush=True)
            return
    except Exception:
        pass

    try:
        print(json.dumps(response_to_dict(resp), ensure_ascii=False, indent=2), flush=True)
        return
    except Exception:
        pass

    print(resp, flush=True)


def extract_answer_from_response(resp: Any) -> str:
    """
    Extrai texto da resposta OpenAI-compatible de forma mais robusta.

    Alguns provedores/modelos podem retornar conteúdo em formatos diferentes:
    - message.content como string
    - message.content como lista de blocos
    - campos alternativos como reasoning_content, text ou output_text
    """

    try:
        choices = getattr(resp, "choices", None)

        if not choices:
            return ""

        choice = choices[0]
        message = getattr(choice, "message", None)

        if message is None:
            return ""

        # Caso padrão: message.content é string
        content = getattr(message, "content", None)

        if isinstance(content, str) and content.strip():
            return content.strip()

        # Caso message.content venha como lista de blocos
        if isinstance(content, list):
            parts = []

            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(str(text))

            if parts:
                return "\n".join(parts).strip()

        # Campos alternativos que alguns modelos/provedores podem usar
        for attr in ["reasoning_content", "text", "output_text"]:
            value = getattr(message, attr, None)

            if isinstance(value, str) and value.strip():
                return value.strip()

        # Fallback usando dict
        data = response_to_dict(message)

        for key in ["content", "reasoning_content", "text", "output_text"]:
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, list):
                parts = []

                for block in value:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content")
                        if text:
                            parts.append(str(text))

                if parts:
                    return "\n".join(parts).strip()

    except Exception:
        return ""

    return ""


def get_finish_reason(resp: Any) -> str:
    try:
        choices = getattr(resp, "choices", None)

        if not choices:
            return ""

        finish_reason = getattr(choices[0], "finish_reason", "")

        if finish_reason:
            return str(finish_reason)

    except Exception:
        pass

    return ""


# ============================================================
# Chamada NIM/OpenAI-compatible
# ============================================================

def nim_chat_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    retries: int = RETRIES,
    base_sleep: float = BASE_SLEEP,
) -> str:
    """
    Chamada padronizada:
    - temperature = 0.1
    - max_tokens = 4096
    - sem top_p explícito
    - sem system prompt
    - prompt enviado como role=user
    - retry também quando a resposta vier vazia
    """

    last_err = None

    for attempt in range(retries):
        try:
            print(
                f"\nChamando modelo {model} | tentativa {attempt + 1}/{retries}",
                flush=True,
            )

            params = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            }

            resp = client.chat.completions.create(
                **params,
                timeout=REQUEST_TIMEOUT_SEC,
            )

            answer = extract_answer_from_response(resp)
            finish_reason = get_finish_reason(resp)

            if not answer.strip():
                print(
                    f"\nResposta vazia recebida de {model}. "
                    f"finish_reason={finish_reason}",
                    flush=True,
                )
                print_raw_response(resp)

                raise RuntimeError(
                    "A API retornou resposta vazia. "
                    "O conteúdo não foi salvo como resposta válida."
                )

            print(
                f"Resposta recebida de {model} com {len(answer)} caracteres "
                f"| finish_reason={finish_reason}",
                flush=True,
            )

            return answer

        except Exception as e:
            last_err = e
            sleep_time = base_sleep * (2 ** attempt)

            print(
                f"\nErro no modelo {model}, tentativa {attempt + 1}/{retries}: {e}",
                flush=True,
            )
            print(
                f"Aguardando {sleep_time:.1f}s antes de tentar novamente...",
                flush=True,
            )

            time.sleep(sleep_time)

    return f"<<ERRO AO GERAR RESPOSTA: {last_err}>>"


# ============================================================
# Execução por modelo
# ============================================================

def run_for_model(
    client: OpenAI,
    model_cfg: dict,
    ds,
    resume: bool = False,
) -> Path:
    model_name = model_cfg["model"]
    alias = model_cfg["alias"]

    print("\n" + "=" * 80)
    print(f"Rodando modelo: {alias}")
    print(f"ID na API: {model_name}")
    print("=" * 80)

    out_file = output_path(alias)

    start_index = load_checkpoint(alias) if resume else 0
    rows = load_existing_rows(alias) if resume else []

    if resume and len(rows) > start_index:
        rows = rows[:start_index]

    for i in tqdm(range(start_index, len(ds)), desc=alias):
        row = ds[i]

        question = row["question"]
        context = row["gold_passage"]
        gold_answer = row["answer"]

        prompt = build_prompt(
            context=context,
            question=question,
        )

        model_answer = nim_chat_completion(
            client=client,
            model=model_name,
            prompt=prompt,
        )

        rows.append(
            {
                "Q": question,
                "A": gold_answer,
                "A_model": model_answer,
            }
        )

        save_checkpoint(alias, i + 1)

        if (i + 1) % SAVE_EVERY == 0:
            save_rows(alias, rows)

        if SLEEP_BETWEEN_CALLS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    save_rows(alias, rows)

    print(f"Arquivo salvo em: {out_file}")

    return out_file


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Número máximo de exemplos para teste.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continua de onde parou usando checkpoint.",
    )

    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Roda apenas modelos cujo alias contenha esse texto.",
    )

    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Carregando dataset...")

    dataset_dict = load_from_disk(DATASET_DIR)

    if SPLIT_NAME not in dataset_dict:
        raise ValueError(
            f"Split '{SPLIT_NAME}' não encontrado. "
            f"Splits disponíveis: {list(dataset_dict.keys())}"
        )

    ds = dataset_dict[SPLIT_NAME]

    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    selected_models = MODELS_TO_RUN

    if args.only:
        selected_models = [
            m for m in MODELS_TO_RUN
            if args.only.lower() in m["alias"].lower()
        ]

    if not selected_models:
        raise RuntimeError(
            f"Nenhum modelo encontrado com filtro --only={args.only}"
        )

    print(f"Total de exemplos: {len(ds)}")
    print(f"Temperatura: {TEMPERATURE}")
    print(f"Max tokens: {MAX_TOKENS}")
    print("Top p: não definido explicitamente")
    print("System prompt: não utilizado")
    print("Modelos selecionados:")

    for model_cfg in selected_models:
        print(f" - {model_cfg['alias']} ({model_cfg['model']})")

    client = get_client()

    start = time.time()
    index_rows = []

    for model_cfg in selected_models:
        out_file = run_for_model(
            client=client,
            model_cfg=model_cfg,
            ds=ds,
            resume=args.resume,
        )

        index_rows.append(
            {
                "provider": "nvidia_nim",
                "model": model_cfg["model"],
                "alias": model_cfg["alias"],
                "file": str(out_file),
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "top_p": "not_set",
                "system_prompt": "none",
                "prompt_template": "presa_like_retriever_template",
            }
        )

        save_index(index_rows)

    elapsed = time.time() - start

    index_file = Path(OUTPUT_DIR) / "index_presa_like_nim.csv"

    print("\nFinalizado.")
    print(f"Tempo total: {elapsed:.2f}s (~{elapsed / 60:.2f} min)")
    print(f"Saídas em: {Path(OUTPUT_DIR).resolve()}")
    print(f"Índice salvo em: {index_file}")


if __name__ == "__main__":
    main()