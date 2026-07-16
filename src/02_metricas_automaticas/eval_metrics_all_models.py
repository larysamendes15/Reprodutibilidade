#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Avalia métricas (ROUGE, BLEU, BERTScore e Connectedness) para todos os CSVs *_QAG.csv.

Cada arquivo *_QAG.csv deve conter as colunas:
  Q        -> pergunta
  A        -> resposta oficial / gold
  A_model  -> resposta gerada pelo modelo

Saídas:
  results/legacy/QAG/aberto/metrics_all_models.csv
  results/legacy/QAG/aberto/<modelo>_metrics.csv

Instalar:
  pip install pandas rouge sacrebleu bert-score sentence-transformers torch
"""

from __future__ import annotations

from pathlib import Path
import re
import random

import numpy as np
import pandas as pd
import torch

from rouge import Rouge
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer


# =========================
# REPRODUTIBILIDADE
# =========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DEVICE
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# CONFIGURAÇÕES
# =========================

INPUT_DIR = Path("results/legacy")
GLOB_PATTERN = "*_QAG.csv"

CSV_OUTPUT_ALL = INPUT_DIR / "metrics_all_models.csv"
SAVE_PER_MODEL = True

# Modelo usado somente para a métrica Connectedness
ST_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


# =========================
# FUNÇÕES AUXILIARES
# =========================

def infer_model_name_from_filename(path: Path) -> str:
    """
    Exemplo:
      llama-3.3-70b-instruct_QAG.csv -> llama-3.3-70b-instruct
    """
    name = path.stem
    name = re.sub(r"_QAG$", "", name)
    return name


def clean_text_list(series: pd.Series) -> list[str]:
    """
    Converte a coluna para lista de strings.
    Caso exista NaN, transforma em string vazia para evitar erro nas métricas.
    """
    return series.fillna("").astype(str).tolist()


def compute_connectedness(
    refs: list[str],
    hyps: list[str],
    st_model: SentenceTransformer,
) -> float:
    """
    Métrica adicionada neste trabalho.

    Calcula a similaridade média entre os embeddings da resposta oficial (A)
    e da resposta gerada pelo modelo (A_model).
    """
    emb_refs = st_model.encode(
        refs,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    emb_hyps = st_model.encode(
        hyps,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    emb_refs = torch.nn.functional.normalize(emb_refs, p=2, dim=1)
    emb_hyps = torch.nn.functional.normalize(emb_hyps, p=2, dim=1)

    scores = (emb_refs * emb_hyps).sum(dim=1)

    return float(scores.mean().item())


def compute_metrics_for_file(
    csv_path: Path,
    rouge: Rouge,
    bleu: BLEU,
    st_model: SentenceTransformer,
) -> dict:
    df = pd.read_csv(csv_path)

    required_columns = ["Q", "A", "A_model"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"[{csv_path.name}] Coluna obrigatória ausente: '{col}'. "
                f"Colunas encontradas: {list(df.columns)}"
            )

    refs = clean_text_list(df["A"])
    hyps = clean_text_list(df["A_model"])

    # =========================
    # ROUGE
    # =========================
    rouge_scores = rouge.get_scores(hyps, refs, avg=True)

    # =========================
    # BLEU
    # =========================
    bleu_score = bleu.corpus_score(hyps, [refs])

    # =========================
    # BERTScore
    # Mantido igual à lógica original:
    # score(hyps, refs, lang="pt", device=..., verbose=True)
    # =========================
    precision, recall, f1 = bert_score(
        hyps,
        refs,
        lang="pt",
        device=DEVICE,
        verbose=True,
    )

    # =========================
    # Connectedness
    # Única métrica adicionada
    # =========================
    connectedness = compute_connectedness(
        refs=refs,
        hyps=hyps,
        st_model=st_model,
    )

    model_name = infer_model_name_from_filename(csv_path)

    return {
        "modelo": model_name,
        "arquivo_qag": csv_path.name,

        "rouge-1": float(rouge_scores["rouge-1"]["f"]),
        "rouge-2": float(rouge_scores["rouge-2"]["f"]),
        "rouge-l": float(rouge_scores["rouge-l"]["f"]),

        "bleu": float(bleu_score.score / 100.0),

        "bert_precision": float(precision.mean().item()),
        "bert_recall": float(recall.mean().item()),
        "bert_f1": float(f1.mean().item()),

        "connectedness": float(connectedness),

        "num_exemplos": int(len(df)),
        "device": DEVICE,
        "seed": SEED,
        "bertscore_lang": "pt",
        "connectedness_model": ST_MODEL_NAME,
    }


# =========================
# MAIN
# =========================

def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {INPUT_DIR.resolve()}"
        )

    files = sorted(INPUT_DIR.glob(GLOB_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo encontrado com padrão '{GLOB_PATTERN}' em "
            f"{INPUT_DIR.resolve()}"
        )

    print(f"Encontrados {len(files)} arquivos para avaliar em: {INPUT_DIR.resolve()}")

    for file in files:
        print(f" - {file.name}")

    rouge = Rouge()
    bleu = BLEU()

    print(f"\nCarregando SentenceTransformer ({ST_MODEL_NAME}) em {DEVICE}...")
    st_model = SentenceTransformer(ST_MODEL_NAME, device=DEVICE)

    all_metrics = []

    for csv_path in files:
        print(f"\nAvaliando: {csv_path.name}")

        metrics = compute_metrics_for_file(
            csv_path=csv_path,
            rouge=rouge,
            bleu=bleu,
            st_model=st_model,
        )

        all_metrics.append(metrics)

        print(
            f"  modelo={metrics['modelo']} | "
            f"N={metrics['num_exemplos']} | "
            f"ROUGE-L={metrics['rouge-l']:.4f} | "
            f"BLEU={metrics['bleu']:.4f} | "
            f"BERT-F1={metrics['bert_f1']:.4f} | "
            f"Connectedness={metrics['connectedness']:.4f} | "
            f"DEVICE={metrics['device']}"
        )

        if SAVE_PER_MODEL:
            output_path = INPUT_DIR / f"{metrics['modelo']}_metrics.csv"
            pd.DataFrame([metrics]).to_csv(
                output_path,
                index=False,
                encoding="utf-8",
            )
            print(f"  -> salvo: {output_path.name}")

    output_df = pd.DataFrame(all_metrics)

    output_df.sort_values(
        by=["bert_f1", "rouge-l", "bleu"],
        ascending=False,
        inplace=True,
    )

    output_df.to_csv(
        CSV_OUTPUT_ALL,
        index=False,
        encoding="utf-8",
    )

    print(f"\n✅ Arquivo agregado salvo em: {CSV_OUTPUT_ALL.resolve()}")
    print("Fim.")


if __name__ == "__main__":
    main()