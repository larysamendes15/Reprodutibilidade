#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QA multi-model via NVIDIA NIM / API Catalog (OpenAI-compatible).

Instalar dependências:
    pip install openai datasets pandas tqdm

Rodar:
    python qa_nim_multi_models.py --max-rows 5
    python qa_nim_multi_models.py
"""

from __future__ import annotations

import os
import time
import json
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
from openai import OpenAI


NIM_API_KEY = os.getenv("NIM_API_KEY", "")

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

DATASET_DIR = "data/tax_law_brazil_cosit"
OUTPUT_DIR = "results/legacy"
SPLIT_NAME = "tax_law"

TEMPERATURE = 0.1
MAX_ROWS: Optional[int] = None
SLEEP_BETWEEN_CALLS_SEC = 0.0

MODELS_TO_RUN = [
    "mistralai/mixtral-8x22b-instruct-v0.1",

]


RETRIEVER_TEMPLATE = """Use the following pieces of legal information from laws to answer the user's question.
If the answer is not clear in context, try to figure out by interpreting the information.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Do not quote the "contextual information" provided in the answer, do not say "according to the information" or anything like that, use the information only to answer the question.
Only return the helpful answer below and nothing else.
REMEMBER: answer the question in portuguese.
Helpful answer:"""

def get_client() -> OpenAI:
    if not NIM_API_KEY or NIM_API_KEY.strip() == "":
        raise RuntimeError("NIM_API_KEY não definida no arquivo.")
    return OpenAI(
        base_url=NIM_BASE_URL,
        api_key=NIM_API_KEY
    )


def build_prompt(context: str, question: str) -> str:
    return RETRIEVER_TEMPLATE.format(context=context, question=question)


def nim_chat_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 1024,
    retries: int = 6,
    base_sleep: float = 1.0,
) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** attempt))
    return f"<<ERRO AO GERAR RESPOSTA: {last_err}>>"

def checkpoint_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return Path(OUTPUT_DIR) / f"{safe}.checkpoint.json"


def load_checkpoint(model_name: str) -> int:
    p = checkpoint_path(model_name)
    if not p.exists():
        return 0
    return json.loads(p.read_text()).get("next_index", 0)


def save_checkpoint(model_name: str, next_index: int):
    p = checkpoint_path(model_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"next_index": next_index}, indent=2))

def run_for_model(client: OpenAI, model_name: str, ds, resume: bool):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_")
    out_file = output_dir / f"{safe_name}_QAG.csv"

    start_index = load_checkpoint(model_name) if resume else 0
    rows = []

    if resume and out_file.exists():
        rows = pd.read_csv(out_file).to_dict(orient="records")

    for i in tqdm(range(start_index, len(ds)), desc=model_name):
        row = ds[i]

        prompt = build_prompt(
            context=row["gold_passage"],
            question=row["question"],
        )

        model_answer = nim_chat_completion(
            client,
            model_name,
            prompt,
            TEMPERATURE
        )

        rows.append({
            "Q": row["question"],
            "A": row["answer"],
            "A_model": model_answer
        })

        save_checkpoint(model_name, i + 1)

        if SLEEP_BETWEEN_CALLS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(out_file, index=False)

    pd.DataFrame(rows).to_csv(out_file, index=False)
    return out_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = get_client()

    dataset_dict = load_from_disk(DATASET_DIR)
    ds = dataset_dict[SPLIT_NAME]

    if args.max_rows:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    print(f"Dataset carregado: {len(ds)} exemplos")
    print("Modelos:")
    for m in MODELS_TO_RUN:
        print(" -", m)

    start = time.time()
    index = []

    for model in MODELS_TO_RUN:
        print(f"\n=== Rodando {model} ===")
        out = run_for_model(client, model, ds, resume=args.resume)
        index.append({"model": model, "file": str(out)})

    pd.DataFrame(index).to_csv(Path(OUTPUT_DIR) / "index.csv", index=False)

    elapsed = time.time() - start
    print(f"\nFinalizado em {elapsed/60:.2f} minutos")


if __name__ == "__main__":
    main()
