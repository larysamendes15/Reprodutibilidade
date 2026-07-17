#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script de QA usando GPT-5.1 (OpenAI).

Estrutura esperada (rodar a partir da raiz do repositório):

    Reprodutibilidade/
        src/01_geracao_respostas/gerar_respostas_gpt.py
        data/tax_law_brazil_cosit/       <- dataset salvo com save_to_disk
        results/respostas_geradas/       <- onde o CSV será gerado

Para rodar:
    pip install openai datasets pandas tqdm
    export OPENAI_API_KEY="sua_chave"
    python src/01_geracao_respostas/gerar_respostas_gpt.py
"""

from pathlib import Path
import time
import os

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "")
DATASET_DIR = "data/tax_law_brazil_cosit"
OUTPUT_DIR = "results/respostas_geradas"
SPLIT_NAME = "tax_law"

MODEL_NAME = "gpt-5.1"

# Número máximo de linhas para processar (None = todas)
MAX_ROWS = None

# Retry com backoff exponencial
MAX_RETRIES = 5
BASE_WAIT_SECONDS = 2.0

SYSTEM_PROMPT = (
    "Você é um assistente jurídico especialista em direito tributário brasileiro. "
    "Use APENAS o contexto fornecido, responda em português, e nunca cite o contexto explicitamente."
)

RETRIEVER_TEMPLATE = """Use the following pieces of legal information from laws to answer the user's question.
If the answer is not clear in context, try to figure out by interpreting the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Do not quote the "contextual information" provided in the answer, do not say "according to the information" or anything like that, use the information only to answer the question.
Only return the helpful answer below and nothing else.
REMEMBER: answer the question in portuguese.
Helpful answer:"""


def configure_gpt(api_key: str) -> OpenAI:
    """Retorna um client OpenAI já validado."""
    real_key = api_key or os.getenv("OPENAI_API_KEY")

    if not real_key:
        raise RuntimeError(
            "Nenhuma API key encontrada. Defina OPENAI_API_KEY."
        )

    return OpenAI(api_key=real_key)


def gpt_infer(client: OpenAI, context: str, question: str) -> str:
    """Chama o GPT-5.1 com o prompt completo, com retry e backoff exponencial.

    Alguns modelos de raciocínio da OpenAI não aceitam `temperature`
    customizada nem o parâmetro legado `max_tokens`; nesses casos a chamada é
    refeita automaticamente sem os parâmetros rejeitados.
    """
    prompt = RETRIEVER_TEMPLATE.format(context=context, question=question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_completion_tokens": 4096,
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()

            if not text:
                raise RuntimeError("Resposta vazia do modelo.")

            return text

        except Exception as e:
            msg = str(e)

            # Modelo não aceita temperature customizada -> remove e tenta de novo
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue

            # SDK/endpoint que ainda espera max_tokens em vez de max_completion_tokens
            if "max_completion_tokens" in msg and "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                continue

            last_error = e
            if attempt < MAX_RETRIES:
                wait = BASE_WAIT_SECONDS * (2 ** (attempt - 1))
                tqdm.write(
                    f"  [retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {msg[:120]} "
                    f"— aguardando {wait:.0f}s"
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Falha após {MAX_RETRIES} tentativas: {last_error}"
    )


def run_experiment() -> None:
    start = time.time()

    client = configure_gpt(API_KEY)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dataset: {DATASET_DIR}")
    ds_dict = load_from_disk(DATASET_DIR)

    if SPLIT_NAME not in ds_dict:
        raise ValueError(f"Split '{SPLIT_NAME}' não encontrado no dataset.")

    ds = ds_dict[SPLIT_NAME]

    if MAX_ROWS is not None:
        ds = ds.select(range(min(MAX_ROWS, len(ds))))

    print(f"Total de exemplos: {len(ds)}")
    print(f"Modelo usado: {MODEL_NAME}")

    rows = []

    for row in tqdm(ds, desc=f"Gerando respostas ({MODEL_NAME})"):
        question = row["question"]
        context = row["gold_passage"]
        gold_answer = row["answer"]

        try:
            model_answer = gpt_infer(client, context, question)
        except Exception as e:
            model_answer = f"<<ERRO: {e}>>"

        rows.append({
            "Q": question,
            "A": gold_answer,
            "A_model": model_answer,
        })

    df = pd.DataFrame(rows)
    out_csv = output_path / f"{MODEL_NAME.replace('/', '_')}_QAG.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\nArquivo salvo em:", out_csv)
    elapsed = time.time() - start
    print(f"Tempo total: {elapsed:.2f}s (~{elapsed/60:.1f} min)")


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
