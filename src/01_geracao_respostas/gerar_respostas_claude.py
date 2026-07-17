#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script de QA usando Claude Sonnet 4.5 (Anthropic).

Estrutura esperada:

    qa_experiment/
        qa_claude.py
        data/tax_law_brazil_cosit/   <- dataset salvo com save_to_disk
        saidas/                 <- onde os CSVs serão gerados

Para rodar:
    pip install anthropic datasets pandas tqdm
    export CLAUDE_API_KEY="sua_chave"
    python qa_claude.py
"""

from pathlib import Path
import time
import os

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from anthropic import Anthropic

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATASET_DIR = "data/tax_law_brazil_cosit"
OUTPUT_DIR = "results/respostas_geradas"
SPLIT_NAME = "tax_law"

MODEL_NAME = "claude-opus-4-6"

MAX_ROWS = None


RETRIEVER_TEMPLATE = """Use the following pieces of legal information from laws to answer the user's question.
If the answer is not clear in context, try to figure out by interpreting the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Do not quote the "contextual information" provided in the answer, do not say "according to the information" or anything like that, use the information only to answer the question.
Only return the helpful answer below and nothing else.
REMEMBER: answer the question in portuguese.
Helpful answer:"""

def configure_claude(api_key: str):
    """Retorna um client Anthropic já validado."""
    real_key = (
        api_key
        or os.getenv("CLAUDE_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )

    if not real_key:
        raise RuntimeError(
            "Nenhuma API key encontrada. Defina CLAUDE_API_KEY ou ANTHROPIC_API_KEY."
        )

    return Anthropic(api_key=real_key)


def claude_infer(client, context: str, question: str) -> str:
    """Chama Claude Sonnet 4.5 com o prompt completo."""
    prompt = RETRIEVER_TEMPLATE.format(context=context, question=question)

    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        temperature=0.1,
        system=(
            "Você é um assistente jurídico especialista em direito tributário brasileiro. "
            "Use APENAS o contexto fornecido, responda em português, e nunca cite o contexto explicitamente."
        ),
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    out = []
    for block in resp.content:
        if block.type == "text":
            out.append(block.text)

    return "".join(out).strip()

def run_experiment():
    start = time.time()

    client = configure_claude(API_KEY)

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
            model_answer = claude_infer(client, context, question)
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


def main():
    run_experiment()


if __name__ == "__main__":
    main()
